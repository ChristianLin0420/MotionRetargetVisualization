# MotionRetargetVisualization

Human motion capture to humanoid robot motion retargeting with smooth, kinematically feasible trajectories.

Built on Levenberg-Marquardt optimization, this tool ensures natural motion transitions while respecting joint limits and maintaining temporal smoothness.

---

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Visualization](#visualization)

---

## Installation

### Create Python Environment

```bash
conda create -n retarget python=3.8
conda activate retarget
pip install -r requirements.txt
```

---

## Data Preparation

### 1. AMASS Dataset

Download [AMASS Dataset](https://amass.is.tue.mpg.de/index.html) with `SMPL+H G` format.

```
MotionRetargetVisualization/
└── data/
    └── AMASS/
        └── AMASS_Complete/
            ├── ACCAD.tar.bz2
            ├── BMLhandball.tar.bz2
            ├── CMU.tar.bz2
            └── ...
```

Extract all archives:

```bash
cd MotionRetargetVisualization/data/AMASS/AMASS_Complete
for file in *.tar.bz2; do tar -xvjf "$file"; done
```

### 2. SMPL Model

Download [SMPL](https://smpl.is.tue.mpg.de/download.php) (pkl format).

```bash
cd MotionRetargetVisualization/data/smpl
unzip SMPL_python_v.1.1.0.zip
```

Rename and move the model files:

```bash
cd SMPL_python_v.1.1.0/smpl/models
mv basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl ../../../SMPL_NEUTRAL.pkl
mv basicmodel_m_lbs_10_207_0_v1.1.0.pkl ../../../SMPL_MALE.pkl
mv basicmodel_f_lbs_10_207_0_v1.1.0.pkl ../../../SMPL_FEMALE.pkl
```

Or create symlinks:

```bash
cd MotionRetargetVisualization/data/smpl
ln -s SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl SMPL_NEUTRAL.pkl
ln -s SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl SMPL_MALE.pkl
ln -s SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl SMPL_FEMALE.pkl
```

### 3. Occlusion Labels

Download [Occlusion Labels](https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc) and place in:

```
MotionRetargetVisualization/data/occlusion/amass_copycat_occlusion_v3.pkl
```

### 4. Text Labels

Extract `resources/texts.zip` to `data/texts/`:

```bash
cd MotionRetargetVisualization
unzip resources/texts.zip -d data/
```

### 5. Robot Model

Place your robot MJCF/URDF files in `resources/robots/<robot_name>/`.

Example for G1 robot:
```
MotionRetargetVisualization/resources/robots/g1/
├── g1_27dof.xml
├── g1_29dof.xml
└── meshes/
```

---

## Usage

### Step 1: Shape Fitting

Fit the SMPL model to match your robot's proportions.

```bash
cd MotionRetargetVisualization/scripts/g1
python fit_robot_shape_g1.py
```

Output: `data/g1/optimized_shape_scale_g1.pkl`

### Step 2: Motion Retargeting

Retarget human motions to your robot.

```bash
cd MotionRetargetVisualization/scripts/g1
python process_humanml3d_g1.py
```

**Options** (edit in script):
- `num_samples = 100` - Number of motions to process (set to `None` for all)

Output: `data/g1/humanml3d_train_retargeted_wholebody_<N>.pkl`

---

## Visualization

### Interactive Viewer (requires display)

```bash
cd MotionRetargetVisualization/scripts
python mujoco_visualization.py
```

### Headless Video Recording

For servers without display, use the video recorder with EGL:

**Basic Recording:**
```bash
python mujoco_video_recorder.py
```

**Multi-Stage Comparison Videos:**

Visualize the full retargeting pipeline:
1. SMPL skeleton (24 joints)
2. Target keypoints (13 joints)  
3. G1 robot (29 DOFs)

```bash
cd MotionRetargetVisualization/scripts

# Side-by-side view (3 panels)
python mujoco_comparison_recorder.py --mode sidebyside --max_motions 10

# Overlay view (all in one)
python mujoco_comparison_recorder.py --mode overlay --max_motions 10

# Process all motions
python mujoco_comparison_recorder.py --mode sidebyside
```

**Command-line Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | `sidebyside` or `overlay` | `overlay` |
| `--max_motions` | Number of videos to generate | `5` |
| `--output_dir` | Custom output directory | `output/videos_comparison_<mode>` |

---

## Output Structure

```
MotionRetargetVisualization/
├── data/
│   └── g1/
│       ├── optimized_shape_scale_g1.pkl    # Shape fitting result
│       └── humanml3d_train_retargeted_*.pkl # Retargeted motions
└── output/
    ├── videos/                              # Basic recordings
    ├── videos_comparison_overlay/           # Overlay comparison videos
    └── videos_comparison_sidebyside/        # Side-by-side comparison videos
```

---

## Data Format

The retargeted motion data (`humanml3d_train_retargeted_*.pkl`) contains:

| Field | Description |
|-------|-------------|
| `root_trans_offset` | Robot root position (T, 3) |
| `dof` | Joint angles (T, 29) |
| `root_rot` | Root orientation quaternion (T, 4) |
| `global_translation` | Robot link positions (T, 30, 3) |
| `mocap_global_translation` | SMPL joint positions (T, 24, 3) |
| `target_keypoints` | IK target keypoints (T, 13, 3) |
| `captions` | Text descriptions |

---

## Troubleshooting

**GLFW Error on headless server:**
```
GLFWError: X11: The DISPLAY environment variable is missing
```
→ Use `mujoco_video_recorder.py` or `mujoco_comparison_recorder.py` instead

**SMPL model not found:**
```
Path ../../data/smpl/SMPL_NEUTRAL.pkl does not exist!
```
→ Follow SMPL setup instructions above (rename or symlink the pkl files)

---

## License

See [LICENSE](LICENSE) for details.
