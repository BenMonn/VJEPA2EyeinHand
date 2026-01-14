# VJEPA2EyeinHand
An attempt to use V-JEPA 2.0 for "eye-in-hand" robotic arm use.

The franka emika panda robotic arm is taken from the Mujoco Menagerie github. Their page can be found here: https://github.com/google-deepmind/mujoco_menagerie 
We recommend you check out their work as well, for they are also doing great things. 

Instructions:
1. python collect_dataset.py
2. python fit_phi_from_dataset.py --data dataset --encoder_ckpt checkpoints/vjepa_target_ep03.pth --epochs 5 --out checkpoints/phi_latent.pth
3. python test_phi_prediction.py
4. python make_goal.py
5. python run_global_vjepa_control.py --model_xml arm_scene.xml --encoder_ckpt checkpoints/vjepa_target_ep03.pth --phi_ckpt checkpoints/phi_latent.pth --goal_img goal.png --steps 600 --H 12 --K 512 --elites 64 --iters 3

Notes:
Run step 5 for more than 600 steps for better and longer learning.

This project is designed to run on ubuntu 22.04 Jammy Jellyfish.
