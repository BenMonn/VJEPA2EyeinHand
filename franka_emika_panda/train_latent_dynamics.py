# train_latent_dynamics.py
# dataset of (I_t, a_t, I_{t+1}) → (z_t, a_t, z_{t+1})
# z_* = h_xi(I_*)
Phi = TinyMLP(input_dim=D + A, output_dim=D)

loss = mse(Phi(concat(z_t, a_t)), z_tp1)
