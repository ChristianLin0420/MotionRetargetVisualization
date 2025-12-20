# MotionRetargetVisualization
SmoothMoCapRetarget converts human motion capture data into smooth, kinematically feasible trajectories for humanoid robots. Built on Levenberg-Marquardt optimization, it ensures natural motion transitions while respecting joint limits and maintaining temporal smoothness across the entire sequence.

## Python Environment

```cmd
conda create -n retarget python=3.8
conda activate retarget
pip install -r requirements.txt
```

## Recourse Preparation

### AMASS Dataset Preparation
Download [AMASS Dataset](https://amass.is.tue.mpg.de/index.html) with `SMPL + H G` format and put it under `MotionRetargetVisualization/data/AMASS/AMASS_Complete/`:
```
|-- MotionRetargetVisualization
   |-- data
      |-- AMASS
         |-- AMASS_Complete 
               |-- ACCAD.tar.bz2
               |-- BMLhandball.tar.bz2
               |-- BMLmovi.tar.bz2
               |-- BMLrub.tar
               |-- CMU.tar.bz2
               |-- ...
               |-- Transitions.tar.bz2

```

And then `cd MotionRetargetVisualization/data/AMASS/AMASS_Complete` extract all the motion files by running:
```
for file in *.tar.bz2; do
    tar -xvjf "$file"
done
```

Then you should have:
```
|-- MotionRetargetVisualization
   |-- data
      |-- AMASS
         |-- AMASS_Complete 
               |-- ACCAD
               |-- BioMotionLab_NTroje
               |-- BMLhandball
               |-- BMLmovi
               |-- CMU
               |-- ...
               |-- Transitions

```

### Occlusion Label Preparation

Download [Occlusion Labels](https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc) and put the `amass_copycat_occlusion_v3.pkl` file under `MotionRetargetVisualization/data/occlusion/`, then you should have:
```
|-- MotionRetargetVisualization
   |-- data
      |-- occlusion
         |-- amass_copycat_occlusion_v3.pkl
```

### Text Label Preparation

Extract `texts.zip` into `MotionRetargetVisualization/data/` directory, and you should have:
```
|-- MotionRetargetVisualization
   |-- data
      |-- texts
         |-- 000000.txt
         |-- 000001.txt
         |-- ...
         |-- 014615.txt
         |-- M000000.txt
         |-- ...
         |-- M014615.txt
```

## SMPL Model Preparation

Download [SMPL](https://smpl.is.tue.mpg.de/download.php) with `pkl` format and put it under `MotionRetargetVisualization/data/smpl/`, and you should have:
```
|-- MotionRetargetVisualization
   |-- data
      |-- smpl
         |-- SMPL_python_v.1.1.0.zip
```

Then `cd MotionRetargetVisualization/data/smpl` and  `unzip SMPL_python_v.1.1.0.zip`, you should have 
```
|-- MotionRetargetVisualization
   |-- data
      |-- smpl
         |-- SMPL_python_v.1.1.0
            |-- models
               |-- basicmodel_f_lbs_10_207_0_v1.1.0.pkl
               |-- basicmodel_m_lbs_10_207_0_v1.1.0.pkl
               |-- basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
            |-- smpl_webuser
            |-- ...
```
Rename these three pkl files and move it under smpl like this:
```
|-- MotionRetargetVisualization
   |-- data
      |-- smpl
         |-- SMPL_FEMALE.pkl
         |-- SMPL_MALE.pkl
         |-- SMPL_NEUTRAL.pkl
```

### Robot Model Preparation

Put the model description of your robot under `MotionRetargetVisualization/resources/robots`.
Take `g1` robot for example, put `g1_29dof.xml` under `MotionRetargetVisualization/resources/robots/g1`

## Retargeting Procedure

### Shape Fitting

Create the directory to store the fitted shape, and run the shape fitting script to align the standard smpl model with your own robot model.
Take `g1` robot for example
- Create the `MotionRetargetVisualization/data/g1` directory
- Run `MotionRetargetVisualization/scripts/g1/fit_robot_shape.ipynb`, the fitted shape will be saved as `MotionRetargetVisualization/data/g1/fit_robot_shape_g1.pkl`

### Retargeting based on Fitted Shape

Run the motion retargeting script.
Take `g1` robot for example
- Run `MotionRetargetVisualization/scripts/g1/process_humanml3d_g1.ipynb`, the retargeted dataset will be saved under `MotionRetargetVisualization/data/g1/`

## Visualization

Run `MotionRetargetVisualization/scripts/mujoco_visualization.py` to visualize the retargeted mocap dataset.