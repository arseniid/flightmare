
#include "flightlib/envs/vision_env/vision_env.hpp"

namespace flightlib {

VisionEnv::VisionEnv()
  : VisionEnv(getenv("FLIGHTMARE_PATH") +
                std::string("/flightpy/configs/vision/config.yaml"),
              0) {}

VisionEnv::VisionEnv(const std::string &cfg_path, const int env_id)
  : EnvBase() {
  // check if configuration file exist
  if (!(file_exists(cfg_path))) {
    logger_.error("Configuration file %s does not exists.", cfg_path);
  }
  // load configuration file
  cfg_ = YAML::LoadFile(cfg_path);
  env_id_ = env_id;
  //
  init();
}

VisionEnv::VisionEnv(const YAML::Node &cfg_node, const int env_id) : EnvBase() {
  cfg_ = cfg_node;
  env_id_ = env_id;

  //
  init();
}

void VisionEnv::init() {
  //
  is_collision_ = false;
  unity_render_offset_ << 0.0, 0.0, 0.0;
  goal_linear_vel_ << 0.0, 0.0, 0.0;
  cmd_.setZeros();

  // create quadrotors
  quad_ptr_ = std::make_shared<Quadrotor>();

  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quad_ptr_->updateDynamics(dynamics);


  // define input and output dimension for the environment
  obs_dim_ = visionenv::kNObs;
  act_dim_ = visionenv::kNAct;
  rew_dim_ = 0;
  num_detected_obstacles_ = visionenv::kNObstacles;

  // load parameters
  loadParam(cfg_);

  // add camera
  if (!configCamera(cfg_)) {
    logger_.error(
      "Cannot config RGB Camera. Something wrong with the config file");
  }

  // checks for taking correct obstacle config
  std::string level_env_path;
  int num_levels = difficulty_levels_.size();
  if (num_envs_ <= num_levels) {
    level_env_path = difficulty_levels_[env_id_] + std::string("/") +
                     env_folder_;
    logger_.info(
      "Only one environment specified: Load configuration from " +
      level_env_path);
  } else {
    if (num_envs_ % num_levels != 0 || num_envs_ > 100 * num_levels) {
      logger_.warn(
        "Either overall number of environments is not fully divisible by "
        "the number of difficulty levels, or too many environments specified! "
        "Some environments might be taken multiple times...");
    }
    level_env_path = difficulty_levels_[(env_id_ / (num_envs_ / num_levels)) % num_levels] +
                     std::string("/") + std::string("environment_") +
                     std::to_string((env_id_ % (num_envs_ / num_levels)) % 100);
  }

  obstacle_cfg_path_ = getenv("FLIGHTMARE_PATH") +
                       std::string("/flightpy/configs/vision/") +
                       level_env_path;

  // add dynamic objects
  std::string dynamic_object_yaml =
    obstacle_cfg_path_ + std::string("/dynamic_obstacles.yaml");
  if (!configDynamicObjects(dynamic_object_yaml)) {
    logger_.error(
      "Cannot config Dynamic Object Yaml. Something wrong with the config "
      "file");
  }

  // add static objects
  static_object_csv_ =
    obstacle_cfg_path_ + std::string("/static_obstacles.csv");
  if (!configStaticObjects(static_object_csv_)) {
    logger_.error(
      "Cannot config Static Object. Something wrong with the config file");
  }

  // use single rotor control or bodyrate control
  Scalar max_force = quad_ptr_->getDynamics().getForceMax();
  Vector<3> max_omega = quad_ptr_->getDynamics().getOmegaMax();
  //
  act_mean_ << (max_force / quad_ptr_->getMass()) / 2, 0.0, 0.0, 0.0;
  act_std_ << (max_force / quad_ptr_->getMass()) / 2, max_omega.x(),
    max_omega.y(), max_omega.z();
}

VisionEnv::~VisionEnv() {}

bool VisionEnv::reset(Ref<Vector<>> obs) {
  quad_state_.setZero();
  pi_act_.setZero();
  old_pi_act_.setZero();
  is_collision_ = false;

  // randomly reset the quadrotor state
  // reset position
  quad_state_.x(QS::POSX) = uniform_dist_(random_gen_);
  quad_state_.x(QS::POSY) = uniform_dist_(random_gen_) * 9.0;
  quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) * 4 + 5.0;

  // reset quadrotor with random states
  quad_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  // use collective thrust and bodyrate control mode
  cmd_.setCmdMode(quadcmd::THRUSTRATE);
  cmd_.collective_thrust = 0;
  cmd_.omega.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool VisionEnv::reset(Ref<Vector<>> obs, bool random) { return reset(obs); }

bool VisionEnv::getObs(Ref<Vector<>> obs) {
  if (obs.size() != obs_dim_) {
    logger_.error("Observation dimension mismatch. %d != %d", obs.size(),
                  obs_dim_);
    return false;
  }
  // compute rotation matrix
  Vector<9> ori = Map<Vector<>>(quad_state_.R().data(), quad_state_.R().size());

  // get N most closest obstacles as the observation
  Vector<visionenv::kNObstacles * visionenv::kNObstaclesState> obstacle_obs;
  Vector<visionenv::kNFreePaths * visionenv::kNFreePathsState> free_paths_obs;
  getObstacleState(obstacle_obs, free_paths_obs);

  // Observations
  obs << goal_linear_vel_, ori, quad_state_.v, obstacle_obs, free_paths_obs;
  return true;
}

bool VisionEnv::getObstacleState(Ref<Vector<>> obs_state,
                                 Ref<Vector<>> free_paths) {
  if (dynamic_objects_.size() <= 0 || static_objects_.size() <= 0) {
    logger_.error("No dynamic or static obstacles.");
    return false;
  }
  // make sure to reset the collision penalty
  relative_pos_norm_.clear();
  obstacle_radius_.clear();

  //
  quad_ptr_->getState(&quad_state_);

  // compute relative distance to dynamic obstacles
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> relative_pos;
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> relative_vel;
  for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
    // compute relative position vector
    Vector<3> delta_pos = dynamic_objects_[i]->getPos() - quad_state_.p;
    relative_pos.push_back(delta_pos);

    // compute first order approximation of obstacle velocity
    relative_vel.push_back(quad_state_.R() * // TODO: Check if velocity transformed correctly into body coordinates
                           (dynamic_objects_[i]->getPos() - dynamic_objects_old_pos_[i]) /
                           sim_dt_);
    dynamic_objects_old_pos_[i] = dynamic_objects_[i]->getPos();

    // compute relative distance
    Scalar obstacle_dist = delta_pos.norm();

    // compute the obstacle radius
    Scalar obs_radius = dynamic_objects_[i]->getScale()[0] / 2;

    if (obstacle_dist - obs_radius <= max_detection_range_) {
      relative_pos_norm_.push_back(obstacle_dist);
      obstacle_radius_.push_back(obs_radius);
    } else {
      relative_pos_norm_.push_back(999.0);
      obstacle_radius_.push_back(0.0);
    }

    //
    if (obstacle_dist < obs_radius) {
      is_collision_ = true;
    }
  }

  // compute relatiev distance to static obstacles
  for (int i = 0; i < (int)static_objects_.size(); i++) {
    // compute relative position vector
    Vector<3> delta_pos = static_objects_[i]->getPos() - quad_state_.p;
    relative_pos.push_back(delta_pos);

    Vector<3> zero_vel = {0.0, 0.0, 0.0};
    relative_vel.push_back(zero_vel);

    // compute relative distance
    Scalar obstacle_dist = delta_pos.norm();

    // compute the obstacle radius
    Scalar obs_radius = static_objects_[i]->getScale()[0] / 2;

    if (obstacle_dist - obs_radius <= max_detection_range_) {
      relative_pos_norm_.push_back(obstacle_dist);
      obstacle_radius_.push_back(obs_radius);
    } else {
      relative_pos_norm_.push_back(999.0);
      obstacle_radius_.push_back(0.0);
    }

    if (obstacle_dist < obs_radius) {
      is_collision_ = true;
    }
  }

  // std::cout << relative_pos_norm_ << std::endl;
  size_t idx = 0;
  std::vector<Scalar> rel_pos_norm_surface = relative_pos_norm_;
  std::transform(relative_pos_norm_.begin(), relative_pos_norm_.end(), obstacle_radius_.begin(), rel_pos_norm_surface.begin(), std::minus<Scalar>());
  std::vector<size_t> indices_sorted = sort_indexes(rel_pos_norm_surface);
  for (size_t sort_idx : indices_sorted) {
    if (idx >= visionenv::kNObstacles) break;

    if (idx < relative_pos.size()) {
      // if enough obstacles in the environment
      if (rel_pos_norm_surface[sort_idx] <= max_detection_range_) {
        // if obstacles are within detection range
        Vector<3> zero_vel = {0.0, 0.0, 0.0};
        // assert correct velocity computation for static and dynamic obstacles
        if (relative_vel[sort_idx] != zero_vel) {
          assert(!(dynamic_objects_[sort_idx % dynamic_objects_.size()]->isStatic()));
        } else {
          assert(static_objects_[sort_idx % dynamic_objects_.size()]->isStatic());
        }
        obs_state.segment<visionenv::kNObstaclesState>(
          idx * visionenv::kNObstaclesState)
          << relative_pos[sort_idx],
          relative_vel[sort_idx],
          obstacle_radius_[sort_idx];
      } else {
        // if obstacles are beyong detection range
        obs_state.segment<visionenv::kNObstaclesState>(
          idx * visionenv::kNObstaclesState)
          << max_detection_range_, max_detection_range_, max_detection_range_,
          0.0, 0.0, 0.0,
          0.0;
      }

    } else {
      // if not enough obstacles in the environment
      obs_state.segment<visionenv::kNObstaclesState>(
        idx * visionenv::kNObstaclesState)
        << max_detection_range_, max_detection_range_, max_detection_range_,
        0.0, 0.0, 0.0,
        0.0;
    }
    idx += 1;
  }

  getPolarVoxel(relative_pos, relative_pos_norm_, obstacle_radius_, indices_sorted, free_paths);

  return true;
}

bool VisionEnv::getPolarVoxel(
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>>& rel_pos_list_B,
  std::vector<Scalar> rel_pos_norm_list,
  std::vector<Scalar> obs_radius_list,
  std::vector<size_t> indices_sorted,
  Ref<Vector<>> polar_voxel) {
    std::vector<Vector<3>> all_free_paths;
    std::vector<Scalar> free_path_lengths;
    // for kNCuts=8: phi, theta angles go from (approx.) -pi/4 to pi/4
    for (int f = -visionenv::kNCuts / 2; f < visionenv::kNCuts / 2; ++f) {
      for (int t = -visionenv::kNCuts / 2; t < visionenv::kNCuts / 2; ++t) {
        Scalar f_cell = (f + 0.5) * (M_PI / visionenv::kNCuts) / 2;
        Scalar t_cell = (t + 0.5) * (M_PI / visionenv::kNCuts) / 2;
        all_free_paths.push_back(getCartesianFromAng(f_cell, t_cell));
        free_path_lengths.push_back(getDistanceToClosestObstacle(rel_pos_list_B, rel_pos_norm_list, obs_radius_list, indices_sorted, f_cell, t_cell));
      }
    }

    size_t amount = 0;
    std::vector<size_t> free_path_indices_sorted = sort_indexes(free_path_lengths);
    // sort descending by the free path lengths
    for (size_t sorted_idx = free_path_indices_sorted.size() - 1; sorted_idx >= 0; sorted_idx--) {
      if (amount >= visionenv::kNFreePaths) {
        break;
      }

      polar_voxel.segment<visionenv::kNFreePathsState>(
        amount * visionenv::kNFreePathsState)
        << all_free_paths[sorted_idx],
           free_path_lengths[sorted_idx];

      amount++;
    }

    return true;
}

Scalar VisionEnv::getDistanceToClosestObstacle(
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>>& rel_pos_list_B,
  std::vector<Scalar> rel_pos_norm_list,
  std::vector<Scalar> obs_radius_list,
  std::vector<size_t> indices_sorted,
  Scalar f_cell, Scalar t_cell) {
    Matrix<3, 3> R = quad_state_.R();
    R.transposeInPlace();
    Vector<3> ray = getCartesianFromAng(f_cell, t_cell);
    Scalar d_min = max_detection_range_;
    for (size_t sort_idx : indices_sorted) {
        Vector<3> rel_pos_W = R * rel_pos_list_B[sort_idx]; // in world coordinates!
        if (rel_pos_norm_list[sort_idx] == 999.0) {
          break; // obstacle too far -> will be mapped anyway to `max_detection_range_`
        }
        Scalar r = obs_radius_list[sort_idx];
        Scalar t_ca = rel_pos_W.dot(ray);
        if (t_ca < 0) {
          continue; // collision behind the ray origin
        }
        Scalar ray_dist = std::sqrt(rel_pos_W.squaredNorm() - t_ca * t_ca);
        if (ray_dist > r) {
          continue; // no collision
        }
        Scalar t_hc = std::sqrt(r * r - ray_dist * ray_dist);
        if (t_ca - t_hc < d_min) {
          d_min = t_ca - t_hc;
          break; // obstacles are sorted by distance -> no other obstacle expected to be closer
        }
    }
    return d_min; // TODO: needs normalization?
}

Vector<3> VisionEnv::getCartesianFromAng(Scalar phi, Scalar theta) {
  Vector<3> cartesian = {
    std::sin(theta) * std::cos(phi),
    std::sin(theta) * std::sin(phi),
    std::cos(theta)
  };
  return cartesian;
}

bool VisionEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs,
                     Ref<Vector<>> reward) {
  if (!act.allFinite() || act.rows() != act_dim_ || rew_dim_ != reward.rows()) {
    return false;
    logger_.error(
      "Cannot run environment simulation. dimension mismatch or invalid "
      "actions.");
  }

  //
  old_pi_act_ = pi_act_;

  // compute actual control actions
  // act has range between [-1, 1] due to Tanh layer of the NN policy
  pi_act_ = act.cwiseProduct(act_std_) + act_mean_;

  cmd_.t += sim_dt_;
  quad_state_.t += sim_dt_;

  // apply old actions to simulate delay
  cmd_.collective_thrust = old_pi_act_(0);
  cmd_.omega = old_pi_act_.segment<3>(1);

  // simulate quadrotor
  quad_ptr_->run(cmd_, sim_dt_);

  // update quadrotor state and old quad_state
  quad_old_state_ = quad_state_;
  quad_ptr_->getState(&quad_state_);

  // simulate dynamic obstacles
  simDynamicObstacles(sim_dt_);

  // update observations
  getObs(obs);

  return computeReward(reward);
}

bool VisionEnv::simDynamicObstacles(const Scalar dt) {
  if (dynamic_objects_.size() <= 0) {
    logger_.warn(
      "No Dynamic Obstacles defined. Skipping dynamic obstacles simulation.");
    return false;
  }
  for (int i = 0; i < int(dynamic_objects_.size()); i++) {
    dynamic_objects_[i]->run(sim_dt_);
  }
  return true;
}

bool VisionEnv::computeReward(Ref<Vector<>> reward) {
  // ---------------------- reward function design
  // - compute collision penalty
  Scalar collision_penalty = 0.0;
  size_t idx = 0;
  for (size_t sort_idx : sort_indexes(relative_pos_norm_)) {
    if (idx >= visionenv::kNObstacles) break;

    Scalar relative_dist =
      (relative_pos_norm_[sort_idx] > 0) &&
          (relative_pos_norm_[sort_idx] < max_detection_range_)
        ? relative_pos_norm_[sort_idx]
        : max_detection_range_;

    const Scalar dist_margin = 0.1;
    if (relative_pos_norm_[sort_idx] <=
        obstacle_radius_[sort_idx] + dist_margin) {
      // compute distance penalty
      collision_penalty += collision_coeff_ * std::exp(-1.0 * relative_dist);
    }

    idx += 1;
  }

  Scalar move_reward =
    move_coeff_ * (quad_state_.p(QS::POSX) - quad_old_state_.p(QS::POSX));

  // - tracking a constant linear velocity
  Scalar lin_vel_reward =
    vel_coeff_ * (quad_state_.v - goal_linear_vel_).norm();

  // - angular velocity penalty, to avoid oscillations
  const Scalar ang_vel_penalty = angular_vel_coeff_ * quad_state_.w.norm();

  //  change progress reward as survive reward
  const Scalar total_reward =
    move_reward + lin_vel_reward + collision_penalty + ang_vel_penalty + survive_rew_;

  // return all reward components for debug purposes
  // only the total reward is used by the RL algorithm
  reward << move_reward, lin_vel_reward, collision_penalty, ang_vel_penalty, survive_rew_,
    total_reward;
  return true;
}

bool VisionEnv::isTerminalState(Scalar &reward) {
  if (is_collision_) {
    reward = -10.0;
    return true;
  }

  // simulation time out
  if (cmd_.t >= max_t_ - sim_dt_) {
    reward = -10.0;
    return true;
  }

  // world boundling box check
  // - x, y, and z
  const Scalar safty_threshold = 0.1;
  bool x_valid = quad_state_.x(QS::POSX) >= world_box_[0] + safty_threshold &&
                 quad_state_.x(QS::POSX) <= world_box_[1] - safty_threshold;
  bool y_valid = quad_state_.x(QS::POSY) >= world_box_[2] + safty_threshold &&
                 quad_state_.x(QS::POSY) <= world_box_[3] - safty_threshold;
  bool z_valid = quad_state_.x(QS::POSZ) >= world_box_[4] + safty_threshold &&
                 quad_state_.x(QS::POSZ) <= world_box_[5] - safty_threshold;
  if (!x_valid || !y_valid || !z_valid) {
    reward = -10.0;
    return true;
  }

  if (quad_state_.x(QS::POSX) >= 60) {
    reward = 10.0;
    return true;
  }
  return false;
}


bool VisionEnv::getQuadAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && pi_act_.allFinite() && (act.size() == pi_act_.size())) {
    act = pi_act_;
    return true;
  }
  return false;
}

bool VisionEnv::getQuadState(Ref<Vector<>> obs) const {
  if (quad_state_.t >= 0.0 && (obs.rows() == visionenv::kNQuadState)) {
    obs << quad_state_.t, quad_state_.p, quad_state_.qx, quad_state_.v,
      quad_state_.w, quad_state_.a, quad_ptr_->getMotorOmega(),
      quad_ptr_->getMotorThrusts();
    return true;
  }
  logger_.error("Get Quadrotor state failed.");
  return false;
}

bool VisionEnv::getDepthImage(Ref<DepthImgVector<>> depth_img) {
  if (!rgb_camera_ || !rgb_camera_->getEnabledLayers()[0]) {
    logger_.error(
      "No RGB Camera or depth map is not enabled. Cannot retrieve depth "
      "images.");
    return false;
  }
  rgb_camera_->getDepthMap(depth_img_);

  depth_img = Map<DepthImgVector<>>((float_t *)depth_img_.data,
                                    depth_img_.rows * depth_img_.cols);
  return true;
}


bool VisionEnv::getImage(Ref<ImgVector<>> img, const bool rgb) {
  if (!rgb_camera_) {
    logger_.error("No Camera! Cannot retrieve Images.");
    return false;
  }

  rgb_camera_->getRGBImage(rgb_img_);

  if (rgb_img_.rows != img_height_ || rgb_img_.cols != img_width_) {
    logger_.error(
      "Image resolution mismatch. Aborting.. Image rows %d != %d, Image cols "
      "%d != %d",
      rgb_img_.rows, img_height_, rgb_img_.cols, img_width_);
    return false;
  }

  if (!rgb) {
    // converting rgb image to gray image
    cvtColor(rgb_img_, gray_img_, CV_RGB2GRAY);
    // map cv::Mat data to Eiegn::Vector
    img = Map<ImgVector<>>(gray_img_.data, gray_img_.rows * gray_img_.cols);
  } else {
    img = Map<ImgVector<>>(rgb_img_.data, rgb_img_.rows * rgb_img_.cols *
                                            rgb_camera_->getChannels());
  }
  return true;
}


bool VisionEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["environment"]) {
    difficulty_levels_ = cfg["environment"]["level"].as<std::vector<std::string>>();
    env_folder_ = cfg["environment"]["env_folder"].as<std::string>();
    world_box_ = cfg["environment"]["world_box"].as<std::vector<Scalar>>();
    std::vector<Scalar> goal_vel_vec =
      cfg["environment"]["goal_vel"].as<std::vector<Scalar>>();
    goal_linear_vel_ = Vector<3>(goal_vel_vec.data());
    max_detection_range_ =
      cfg["environment"]["max_detection_range"].as<Scalar>();
  }

  if (cfg["simulation"]) {
    sim_dt_ = cfg["simulation"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["simulation"]["max_t"].as<Scalar>();
    num_envs_ = cfg["simulation"]["num_envs"].as<int>();
  } else {
    logger_.error("Cannot load [quadrotor_env] parameters");
    return false;
  }

  if (cfg["rewards"]) {
    // load reward coefficients for reinforcement learning
    move_coeff_ = cfg["rewards"]["move_coeff"].as<Scalar>();
    vel_coeff_ = cfg["rewards"]["vel_coeff"].as<Scalar>();
    collision_coeff_ = cfg["rewards"]["collision_coeff"].as<Scalar>();
    angular_vel_coeff_ = cfg["rewards"]["angular_vel_coeff"].as<Scalar>();
    survive_rew_ = cfg["rewards"]["survive_rew"].as<Scalar>();

    // load reward settings
    reward_names_ = cfg["rewards"]["names"].as<std::vector<std::string>>();
    rew_dim_ = reward_names_.size();
  } else {
    logger_.error("Cannot load [rewards] parameters");
    return false;
  }

  // environment
  if (cfg["unity"]) {
    unity_render_ = cfg["unity"]["render"].as<bool>();
    scene_id_ = cfg["unity"]["scene_id"].as<SceneID>();
  }

  //
  std::string scene_file =
    getenv("FLIGHTMARE_PATH") + std::string("/flightpy/configs/scene.yaml");
  // check if configuration file exist
  if (!(file_exists(scene_file))) {
    logger_.error("Unity scene configuration file %s does not exists.",
                  scene_file);
  }
  // load configuration file
  YAML::Node scene_cfg_node = YAML::LoadFile(scene_file);
  std::string scene_idx = "scene_" + std::to_string(scene_id_);

  std::vector<Scalar> render_offset =
    scene_cfg_node[scene_idx]["render_offset"].as<std::vector<Scalar>>();
  unity_render_offset_ = Vector<3>(render_offset.data());
  return true;
}

bool VisionEnv::configDynamicObjects(const std::string &yaml_file) {
  //
  if (!(file_exists(yaml_file))) {
    logger_.error("Configuration file %s does not exists.", yaml_file);
    return false;
  }
  YAML::Node cfg_node = YAML::LoadFile(yaml_file);

  // logger_.info("Configuring dynamic objects");

  int num_objects = cfg_node["N"].as<int>();
  // create static objects
  for (int i = 0; i < num_objects; i++) {
    std::string object_id = "Object" + std::to_string(i + 1);
    std::string prefab_id = cfg_node[object_id]["prefab"].as<std::string>();
    std::shared_ptr<UnityObject> obj =
      std::make_shared<UnityObject>(object_id, prefab_id);

    // load location, rotation and size
    std::vector<Scalar> posvec =
      (cfg_node[object_id]["position"]).as<std::vector<Scalar>>();
    std::vector<Scalar> rotvec =
      (cfg_node[object_id]["rotation"]).as<std::vector<Scalar>>();
    std::vector<Scalar> scalevec =
      (cfg_node[object_id]["scale"]).as<std::vector<Scalar>>();

    obj->setPosition(Vector<3>(posvec.data()));
    obj->setRotation(Quaternion(rotvec.data()));
    // actual size in meters
    obj->setSize(Vector<3>(1.0, 1.0, 1.0));
    // scale of the original size
    obj->setScale(Vector<3>(scalevec.data()));

    std::string csv_name = cfg_node[object_id]["csvtraj"].as<std::string>();
    std::string csv_file = obstacle_cfg_path_ + std::string("/csvtrajs/") +
                           csv_name + std::string(".csv");
    if (!(file_exists(csv_file))) {
      logger_.error("Configuration file %s does not exists.", csv_file);
      return false;
    }
    obj->loadTrajectory(csv_file);

    dynamic_objects_.push_back(obj);
  }

  for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
    dynamic_objects_old_pos_.push_back(dynamic_objects_[i]->getPos());
  }

  num_dynamic_objects_ = dynamic_objects_.size();
  return true;
}

bool VisionEnv::configStaticObjects(const std::string &csv_file) {
  //
  if (!(file_exists(csv_file))) {
    logger_.error("Configuration file %s does not exists.", csv_file);
    return false;
  }
  std::ifstream infile(csv_file);
  int i = 0;
  for (auto &row : CSVRange(infile)) {
    // Read column 0 for time
    std::string object_id = "StaticObject" + std::to_string(i + 1);
    std::string prefab_id = (std::string)row[0];

    //
    std::shared_ptr<UnityObject> obj =
      std::make_shared<UnityObject>(object_id, prefab_id);

    //
    Vector<3> pos;
    pos << std::stod((std::string)row[1]), std::stod((std::string)row[2]),
      std::stod((std::string)row[3]);

    Quaternion quat;
    quat.w() = std::stod((std::string)row[4]);
    quat.x() = std::stod((std::string)row[5]);
    quat.y() = std::stod((std::string)row[6]);
    quat.z() = std::stod((std::string)row[7]);

    Vector<3> scale;
    scale << std::stod((std::string)row[8]), std::stod((std::string)row[9]),
      std::stod((std::string)row[10]);

    //
    obj->setPosition(pos);
    obj->setRotation(quat);
    // actual size in meters
    obj->setSize(Vector<3>(1.0, 1.0, 1.0));
    // scale of the original size
    obj->setScale(scale);
    static_objects_.push_back(obj);
  }
  num_static_objects_ = static_objects_.size();

  return true;
}

bool VisionEnv::configCamera(const YAML::Node &cfg) {
  if (!cfg["rgb_camera"]) {
    logger_.error("Cannot config RGB Camera");
    return false;
  }

  if (!cfg["rgb_camera"]["on"].as<bool>()) {
    logger_.warn("Camera is off. Please turn it on.");
    return false;
  }

  if (quad_ptr_->getNumCamera() >= 1) {
    logger_.warn("Camera has been added. Skipping the camera configuration.");
    return false;
  }

  // create camera
  rgb_camera_ = std::make_shared<RGBCamera>();

  // load camera settings
  std::vector<Scalar> t_BC_vec =
    cfg["rgb_camera"]["t_BC"].as<std::vector<Scalar>>();
  std::vector<Scalar> r_BC_vec =
    cfg["rgb_camera"]["r_BC"].as<std::vector<Scalar>>();

  //
  Vector<3> t_BC(t_BC_vec.data());
  Matrix<3, 3> r_BC =
    (AngleAxis(r_BC_vec[2] * M_PI / 180.0, Vector<3>::UnitZ()) *
     AngleAxis(r_BC_vec[1] * M_PI / 180.0, Vector<3>::UnitY()) *
     AngleAxis(r_BC_vec[0] * M_PI / 180.0, Vector<3>::UnitX()))
      .toRotationMatrix();
  std::vector<bool> post_processing = {false, false, false};
  post_processing[0] = cfg["rgb_camera"]["enable_depth"].as<bool>();
  post_processing[1] = cfg["rgb_camera"]["enable_segmentation"].as<bool>();
  post_processing[2] = cfg["rgb_camera"]["enable_opticalflow"].as<bool>();

  //
  rgb_camera_->setFOV(cfg["rgb_camera"]["fov"].as<Scalar>());
  rgb_camera_->setWidth(cfg["rgb_camera"]["width"].as<int>());
  rgb_camera_->setChannels(cfg["rgb_camera"]["channels"].as<int>());
  rgb_camera_->setHeight(cfg["rgb_camera"]["height"].as<int>());
  rgb_camera_->setRelPose(t_BC, r_BC);
  rgb_camera_->setPostProcessing(post_processing);


  // add camera to the quadrotor
  quad_ptr_->addRGBCamera(rgb_camera_);

  // adapt parameters
  img_width_ = rgb_camera_->getWidth();
  img_height_ = rgb_camera_->getHeight();
  rgb_img_ = cv::Mat::zeros(img_height_, img_width_,
                            CV_MAKETYPE(CV_8U, rgb_camera_->getChannels()));
  depth_img_ = cv::Mat::zeros(img_height_, img_width_, CV_32FC1);
  return true;
}

bool VisionEnv::addQuadrotorToUnity(const std::shared_ptr<UnityBridge> bridge) {
  if (!quad_ptr_) return false;
  bridge->addQuadrotor(quad_ptr_);

  for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
    bridge->addDynamicObject(dynamic_objects_[i]);
  }

  //
  bridge->setRenderOffset(unity_render_offset_);
  bridge->setObjectCSV(static_object_csv_);
  return true;
}

bool VisionEnv::setUnity(bool render) {
  unity_render_ = render;
  if (!unity_render_ || unity_bridge_ptr_ != nullptr) {
    logger_.warn(
      "Unity render is False or Flightmare Bridge has been already created. "
      "Cannot set Unity.");
    return false;
  }
  // create unity bridge
  unity_bridge_ptr_ = UnityBridge::getInstance();
  // add objects to Unity

  addQuadrotorToUnity(unity_bridge_ptr_);

  logger_.info("Flightmare Bridge created.");
  return true;
}


bool VisionEnv::connectUnity(void) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}


FrameID VisionEnv::updateUnity(const FrameID frame_id) {
  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(frame_id);
    return unity_bridge_ptr_->handleOutput(frame_id);
  } else {
    return 0;
  }
}


void VisionEnv::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

int VisionEnv::getNumDetectedObstacles(void) { return num_detected_obstacles_; }

std::ostream &operator<<(std::ostream &os, const VisionEnv &vision_env) {
  os.precision(3);
  os << "Vision Environment:\n"
     << "obs dim =            [" << vision_env.obs_dim_ << "]\n"
     << "act dim =            [" << vision_env.act_dim_ << "]\n"
     << "#dynamic objects=    [" << vision_env.num_dynamic_objects_ << "]\n"
     << "#static objects=     [" << vision_env.num_static_objects_ << "]\n"
     << "obstacle dim =       [" << vision_env.num_detected_obstacles_ << "]\n"
     << "sim dt =             [" << vision_env.sim_dt_ << "]\n"
     << "max_t =              [" << vision_env.max_t_ << "]\n"
     << "act_mean =           [" << vision_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << vision_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << vision_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << vision_env.obs_std_.transpose() << "]"
     << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib