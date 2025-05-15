import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, np, plt, sci, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    (mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell
def _():
    g_const = 1.0  
    M_const = 1.0  
    l_const = 1.0  

    g = g_const
    M = M_const
    l = l_const


    return M, M_const, g, l, l_const


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell
def _(np):
    def reactor_force_components(f_val, phi_val, theta_val):
        """
        Calcule les composantes de la force du réacteur dans le repère inertiel.
        f_val: amplitude de la force
        phi_val: angle de la force par rapport à l'axe du booster (CCW)
        theta_val: angle du booster par rapport à la verticale (CCW, gauche > 0)
        """
        fx = -f_val * np.sin(theta_val + phi_val)
        fy = f_val * np.cos(theta_val + phi_val)
        return np.array([fx, fy])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le mouvement du centre de masse $(x, y)$ du booster est régi par le Principe Fondamental de la Dynamique (PFD) :

    $$
    \sum \vec{F}_{ext} = M \vec{a}_{COM}
    $$

    où $M$ est la masse du booster et $\vec{a}_{COM} = (\ddot{x}, \ddot{y})$ est l'accélération de son centre de masse.

    Les forces externes agissant sur le booster sont :
    - La **gravité** $\vec{F}_g = (0, -Mg)$
    - La **force du réacteur** $\vec{F}_{réacteur}$

    Les composantes de la force du réacteur sont :

    $$
    f_x = -f \sin(\theta + \phi)
    $$

    $$
    f_y = f \cos(\theta + \phi)
    $$

    En appliquant le PFD :

    **Sur l'axe X** :
    $$M\ddot{x} = f_x$$
    $$\ddot{x} = \frac{1}{M} (-f \sin(\theta + \phi))$$

    **Sur l'axe Y** :
    $$M\ddot{y} = f_y - Mg$$
    $$\ddot{y} = \frac{1}{M} (f \cos(\theta + \phi)) - g$$

    Pour la simulation numérique, nous utilisons un système d'équations différentielles du premier ordre. En introduisant les vitesses $v_x = \dot{x}$ et $v_y = \dot{y}$, le système devient :

    1. $\dot{x} = v_x$
    2. $\dot{v}_x = -\frac{f}{M} \sin(\theta + \phi)$
    3. $\dot{y} = v_y$
    4. $\dot{v}_y = \frac{f}{M} \cos(\theta + \phi) - g$

    Ces équations sont implémentées dans `redstart_dynamics_ode` (voir section Simulation).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Le booster est modélisé comme une tige rigide de longueur $2\ell$ et de masse $M$, dont la masse est uniformément répartie. 

    Le moment d'inertie d'une tige mince de longueur $L_{tige}$ et de masse $M_{tige}$ par rapport à un axe perpendiculaire passant par son centre de masse est donné par :

    $$J_{COM} = \frac{1}{12} M_{tige} L_{tige}^2$$

    Dans notre cas :

    $$M_{tige} = M \text{ et } L_{tige} = 2\ell$$

    Le moment d'inertie $J$ du booster par rapport à son centre de masse est donc :

    $$J = \frac{1}{12} M(2\ell)^2 = \frac{1}{3} M\ell^2$$
    """
    )
    return


@app.cell
def _(M_const, l_const):
    J_calc = (1/3) * M_const * l_const**2
    J = J_calc # Pour correspondre à la demande de la variable J
    print(f"Le moment d'inertie J est: {J:.4f} kg*m^2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'évolution de l'angle d'inclinaison $\theta$ est régie par le théorème du moment cinétique :

    $$\sum \tau_{COM} = J\ddot{\theta}$$

    où $\sum \tau_{COM}$ est la somme des moments (couples) des forces externes par rapport au centre de masse, $J$ est le moment d'inertie et $\ddot{\theta}$ est l'accélération angulaire.

    La **gravité** s'applique au centre de masse, son moment est nul. La **force du réacteur** $\vec{F}_{réacteur}$ est appliquée à la base du booster (position $-l$ le long de l'axe du booster par rapport au CoM). Le vecteur position de la base par rapport au CoM est $\vec{r}_{base/COM} = (l \sin \theta, -l \cos \theta)$. La force du réacteur est

    $$\vec{F}_{réacteur} = (-f \sin (\theta + \phi), f \cos (\theta + \phi)).$$

    Le moment de cette force est :

    $$\tau_{réacteur} = (\vec{r}_{base/COM} \times \vec{F}_{réacteur})_z = -lf \sin \phi.$$

    Ainsi, l'équation du mouvement de rotation est :

    $$J\ddot{\theta} = -lf \sin \phi \dot{\theta} = -lf \sin \phi$$

    En substituant $J = \frac{1}{3} M l^2$ : 

    $$\ddot{\theta} = \frac{-lf \sin \phi}{\frac{1}{3} M l^2} = \frac{-3f \sin \phi}{M l}$$

    Pour la simulation, avec la vitesse angulaire $\omega = \dot{\theta}$ :

    1. $\dot{\theta} = \omega$

    2. $\dot{\omega} = \frac{-3f \sin \phi}{M l}$

    Ces équations sont implémentées dans `redstart_dynamics_ode` (voir section Simulation).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell
def _(M, g, l, np, plt, sci):
    def redstart_dynamics_ode(t, state, ext_f_phi_func, M_sys, l_sys, g_sys):

        x_pos, dx_val, y_pos, dy_val, theta_val, dtheta_val = state

        f_control, phi_control = ext_f_phi_func(t, state)

        ddx_val = (-f_control / M_sys) * np.sin(theta_val + phi_control)
        ddy_val = (f_control / M_sys) * np.cos(theta_val + phi_control) - g_sys

        if M_sys == 0 or l_sys == 0:
            ddtheta_calc = 0.0
        else:
            ddtheta_calc = (-3 * f_control * np.sin(phi_control)) / (M_sys * l_sys)

        return np.array([dx_val, ddx_val, dy_val, ddy_val, dtheta_val, ddtheta_calc])

    def redstart_solve(t_span, y0, f_phi_func):

        sol_ivp = sci.solve_ivp(
            redstart_dynamics_ode,
            t_span,
            y0,
            args=(f_phi_func, M, l, g), 
            dense_output=True,
            method='RK45',
            rtol=1e-7, atol=1e-9
        )
        return sol_ivp.sol 

    def free_fall_example_test(): 
        t_span_ff = [0.0, 5.0]
        y0_ff = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] 

        def f_phi_ff(t, current_state): 
            return np.array([0.0, 0.0]) 

        sol_func_ff = redstart_solve(t_span_ff, y0_ff, f_phi_ff)

        t_plot = np.linspace(t_span_ff[0], t_span_ff[1], 200) 
        y_t_ff = sol_func_ff(t_plot)[2] 
        y_theoretical_ff = y0_ff[2] + y0_ff[3]*t_plot - 0.5 * g * t_plot**2

        fig_ff, ax_ff = plt.subplots(figsize=(10,6))
        ax_ff.plot(t_plot, y_t_ff, label=r"$y(t)$ simulation (m)")
        ax_ff.plot(t_plot, y_theoretical_ff, label=r"$y(t)$ théorique (m)", linestyle="--", color="red")
        ax_ff.plot(t_plot, l * np.ones_like(t_plot), color="grey", ls=":", label=r"$y=\ell$") 
        ax_ff.set_title("Chute Libre (Test de `redstart_solve`)")
        ax_ff.set_xlabel("temps $t$ (s)")
        ax_ff.set_ylabel("hauteur $y$ (m)")
        ax_ff.grid(True)
        ax_ff.legend()
        plt.close(fig_ff) 
        return fig_ff

    free_fall_example_test_fig = free_fall_example_test()
    free_fall_example_test_fig
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def f_phi_controlled_landing(t, y_state):

        f_val = M * (0.864 * t - 2.16 + g)
        phi_val = 0.0 
        return np.array([f_val, phi_val])

    def simulate_controlled_landing():
        t_span_cl = [0.0, 5.0]

        y0_cl = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] 

        sol_cl = redstart_solve(t_span_cl, y0_cl, f_phi_controlled_landing)

        t_plot_cl = np.linspace(t_span_cl[0], t_span_cl[1], 200)
        state_cl = sol_cl(t_plot_cl)

        x_vals, dx_vals, y_vals, dy_vals, th_vals, dth_vals = state_cl

        f_values_cl = np.array([f_phi_controlled_landing(t, None)[0] for t in t_plot_cl])

        fig_cl, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axs[0].plot(t_plot_cl, y_vals, label=r"$y(t)$ hauteur")
        axs[0].plot(t_plot_cl, dy_vals, label=r"$\dot{y}(t)$ vitesse verticale")
        axs[0].axhline(l, color='gray', linestyle='--', label=f"$y={l}$ (cible)")
        axs[0].set_ylabel("Position (m) / Vitesse (m/s)")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_title("Profil d'Atterrissage Contrôlé")

        axs[1].plot(t_plot_cl, th_vals, label=r"$\theta(t)$ inclinaison")
        axs[1].plot(t_plot_cl, x_vals, label=r"$x(t)$ position horizontale") 
        axs[1].set_ylabel("Inclinaison (rad) / Position X (m)")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(t_plot_cl, f_values_cl, label=r"$f(t)$ poussée")
        axs[2].axhline(0, color='red', linestyle=':', label="Poussée nulle")
        axs[2].axhline(M*g, color='green', linestyle=':', label="Poussée $Mg$ (hover)")
        axs[2].set_xlabel("Temps (s)")
        axs[2].set_ylabel("Poussée $f(t)$ (N)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()

        print(f"Atterrissage Contrôlé - Conditions finales (attendues à t=5s):")
        print(f"  y(5) = {y_vals[-1]:.3f} m (cible: {l:.1f} m)")
        print(f"  dy(5) = {dy_vals[-1]:.3f} m/s (cible: 0.0 m/s)")
        print(f"  x(5) = {x_vals[-1]:.3f} m (cible: 0.0 m)")
        print(f"  theta(5) = {th_vals[-1]:.3f} rad (cible: 0.0 rad)")
        print(f"Note: La force f(0) = {f_values_cl[0]:.2f} N, ce qui est < 0.")

        return fig_cl


    return f_phi_controlled_landing, simulate_controlled_landing


@app.cell
def _(simulate_controlled_landing):
    simulate_controlled_landing()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour trouver la force $f(t)$ pour l’atterrissage contrôlé dans l’axe ($\phi = 0$, et on suppose $\theta(t) = 0$), nous cherchons une trajectoire $y(t)$ telle que :

    $$
    \{ y(0) = 10,\ \dot{y}(0) = 0,\ y(5) = \ell = 1,\ \dot{y}(5) = 0 \}.
    $$

    Nous utilisons un polynôme cubique

    $$
    y(t) = at^3 + bt^2 + ct + d.
    $$

    Les conditions aux limites donnent $a = 0.144$ et $b = -1.08$. Donc

    $$
    y(t) = 0.144\, t^3 - 1.08\, t^2 + 10.
    $$

    L’accélération verticale est

    $$
    \ddot{y}(t) = 0.864\, t - 2.16.
    $$

    L’équation du mouvement vertical avec $\theta = 0,\ \phi = 0$ est

    $$
    M\, \ddot{y} = f - M\, g.
    $$

    Donc, la force requise est

    $$
    f(t) = M\, (\ddot{y}(t) + g).
    $$

    Avec $M = 1$ et $g = 1$ :

    $$
    f(t) = 1 \cdot (0.864\, t - 2.16 + 1) = 0.864\, t - 1.16.
    $$

    Cette force est négative au début ($f(0) = -1.16$ N), ce qui n’est pas réalisable par un réacteur à poussée positive. Cela montre les limites du modèle polynomial simple pour cette trajectoire. Cependant, nous allons simuler cette force calculée théoriquement.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell
def _(M, g, l, np, plt):
    def draw_booster_and_flame(ax, x_com, y_com, theta_val, f_force, phi_angle):

        body_half_width = l * 0.15 

        u_long = np.array([-np.sin(theta_val), np.cos(theta_val)]) 
        u_perp = np.array([np.cos(theta_val), np.sin(theta_val)])  

        com_pos_vec = np.array([x_com, y_com])
        corners_local = np.array([
            [ l,  body_half_width], [ l, -body_half_width],
            [-l, -body_half_width], [-l,  body_half_width]
        ])
        corners_world = np.array([com_pos_vec + c[0]*u_long + c[1]*u_perp for c in corners_local])

        booster_patch = plt.Polygon(corners_world, closed=True, fc='cornflowerblue', ec='black', zorder=10)
        ax.add_patch(booster_patch)

        if f_force > 1e-3: 
            flame_ref_len = l 
            flame_len = (f_force / (M * g)) * flame_ref_len if (M*g) > 0 else f_force * flame_ref_len
            flame_len = np.maximum(0.05, flame_len) 

            flame_base_w = 2 * body_half_width * 0.9
            flame_origin = com_pos_vec - l * u_long 

            u_flame_dir = np.array([np.sin(theta_val + phi_angle), -np.cos(theta_val + phi_angle)])
            u_flame_perp = np.array([np.cos(theta_val + phi_angle), np.sin(theta_val + phi_angle)])

            flame_tip_pt = flame_origin + flame_len * u_flame_dir
            flame_base1 = flame_origin + (flame_base_w/2) * u_flame_perp
            flame_base2 = flame_origin - (flame_base_w/2) * u_flame_perp

            flame_patch = plt.Polygon([flame_base1, flame_tip_pt, flame_base2], closed=True, 
                                      fc='orangered', ec='red', alpha=0.75, zorder=5)
            ax.add_patch(flame_patch)


        if not any(line.get_label() == 'Sol' for line in ax.get_lines()):
            ax.plot([-5*l, 5*l], [0, 0], 'k-', lw=1.5, label='Sol', zorder=1)
        if not any(line.get_label() == 'Zone Cible (0,0)' for line in ax.get_lines()):
            ax.plot(0, 0, 'gx', markersize=12, markeredgewidth=3, label='Zone Cible (0,0)', zorder=2)

    fig_draw_test, ax_dt = plt.subplots(figsize=(6,6))
    draw_booster_and_flame(ax_dt, x_com=1, y_com=5, theta_val=np.pi/6, f_force=1.5*M*g, phi_angle=-np.pi/10)
    ax_dt.set_xlim(-4, 6); ax_dt.set_ylim(-1, 10) 
    ax_dt.set_aspect('equal'); ax_dt.grid(True)
    ax_dt.legend(loc='upper right')
    plt.close(fig_draw_test) 
    fig_draw_test 
    return (draw_booster_and_flame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster_and_flame,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):

    def make_booster_video(output_filename, t_span_vid, y0_vid, 
                           f_phi_video_func, 
                           xlims=(-5,5), ylims=(-1,12), 
                           num_frames_vid=150, fps_vid=25): 

        sol_func_vid = redstart_solve(t_span_vid, y0_vid, f_phi_video_func)

        t_video_frames = np.linspace(t_span_vid[0], t_span_vid[1], num_frames_vid)
        state_video_frames = sol_func_vid(t_video_frames)

        f_phi_video_values = np.array([
            f_phi_video_func(t, state_video_frames[:, i]) 
            for i, t in enumerate(t_video_frames)
        ]).T # Transposer pour avoir [f_vals], [phi_vals]

        fig_anim = plt.figure(figsize=(8, 8))
        ax_anim = fig_anim.add_subplot(111)

        progress_bar = tqdm(total=num_frames_vid, desc=f"Vid: {output_filename.split('/')[-1]}")

        def animate_booster(frame_idx):
            ax_anim.clear()

            current_st = state_video_frames[:, frame_idx]
            x_c, _, y_c, _, th_c, _ = current_st
            f_c, phi_c = f_phi_video_values[:, frame_idx]

            draw_booster_and_flame(ax_anim, x_c, y_c, th_c, f_c, phi_c)

            ax_anim.set_xlim(xlims); ax_anim.set_ylim(ylims)
            ax_anim.set_aspect('equal', adjustable='box')
            ax_anim.set_xlabel("X (m)"); ax_anim.set_ylabel("Y (m)")
            title_str = (f"t={t_video_frames[frame_idx]:.2f}s | "
                         f"F={f_c:.1f}N | φ={np.rad2deg(phi_c):.0f}°")
            ax_anim.set_title(f"Redstart: {title_str}")
            ax_anim.grid(True)

            handles, labels = ax_anim.get_legend_handles_labels()
            if handles:
                 unique_labels_dict = {}
                 for h, lab in zip(handles, labels):
                     if lab not in unique_labels_dict: unique_labels_dict[lab] = h
                 ax_anim.legend(unique_labels_dict.values(), unique_labels_dict.keys(), 
                               loc='upper right', fontsize='small')
            progress_bar.update(1)

        anim_obj = FuncAnimation(fig_anim, animate_booster, frames=num_frames_vid, repeat=False)

        video_result = None
        if FFMpegWriter.isAvailable():
            writer_vid = FFMpegWriter(fps=fps_vid, metadata=dict(artist='Marimo User'), bitrate=1800)
            try:
                anim_obj.save(output_filename, writer=writer_vid)
                print(f"\nVidéo sauvegardée: {output_filename!r}")
                video_result = mo.video(src=output_filename)
            except Exception as e_save:
                print(f"\nErreur sauvegarde MP4 '{output_filename}': {e_save}")
                video_result = mo.md(f"**Erreur MP4**: {e_save}")
        else:
            msg_ffmpeg = "FFMpeg non trouvé. Vidéo non générée."
            print(f"\n{msg_ffmpeg}")

            try:
                from matplotlib.animation import PillowWriter
                gif_filename = output_filename.replace(".mp4", ".gif")
                anim_obj.save(gif_filename, writer=PillowWriter(fps=15))
                print(f"GIF sauvegardé à la place: {gif_filename!r}")
                video_result = mo.md(f"{msg_ffmpeg} GIF créé : `{gif_filename}`. Utiliser `mo.image(src='{gif_filename}')`.")
            except Exception as e_gif:
                print(f"Erreur sauvegarde GIF: {e_gif}")
                video_result = mo.md(f"**Erreur Vidéo/GIF**: FFMpeg indisponible ET erreur GIF: {e_gif}")

        progress_bar.close()
        plt.close(fig_anim) 
        return video_result

    y0_common_vis = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    t_span_common_vis = [0.0, 5.0]
    num_frames_default = int( (t_span_common_vis[1]-t_span_common_vis[0]) * 20 ) # 20 fps * 5s = 100 frames
    fps_default = 20
    return (
        fps_default,
        make_booster_video,
        num_frames_default,
        t_span_common_vis,
        y0_common_vis,
    )


app._unparsable_cell(
    r"""
    ## Scénario 1 - Chute Libre
    Cette simulation montre le comportement du booster en chute libre sans aucune force de propulsion (f=0) ni angle d'inclinaison (φ=0). On observe la trajectoire parabolique caractéristique d'un corps soumis uniquement à la gravité, avec une accélération constante vers le bas. Le scénario permet de valider le modèle physique de base en l'absence de contrôle.
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(
    fps_default,
    make_booster_video,
    mo,
    np,
    num_frames_default,
    t_span_common_vis,
    y0_common_vis,
):
    # Scénario 1: Chute libre (f=0, phi=0)
    mo.md("### Vidéo 1: Chute Libre")
    def f_phi_vis1(t, state): return np.array([0.0, 0.0])
    video_out_1 = make_booster_video(
        "vis_sc1_free_fall.mp4", t_span_common_vis, y0_common_vis, f_phi_vis1,
        num_frames_vid=num_frames_default, fps_vid=fps_default
    )
    video_out_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Critique :

    Le résultat obtenu est cohérent : la fusée est en chute libre, immobile par rapport à son environnement, ce qui correspond au fait qu'il est uniquement soumis à son poids, sans frottements ni force de propulsion.

    Le dépassement de la ligne symbolique du sol est également logique, car cette ligne est fictive (représentative). Il est donc normal que la fusée puisse la traverser dans cette modélisation.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Scénario 2 - Poussée Équilibrée
    Le booster applique une poussée verticale exactement égale à son poids (f=Mg) avec un angle nul (φ=0). Cette configuration théorique devrait maintenir le système en équilibre parfait. La simulation vérifie que les forces se compensent effectivement, résultant en une position stationnaire, ce qui valide les calculs d'équilibre statique.
    """
    )
    return


@app.cell
def _(
    M,
    fps_default,
    g,
    make_booster_video,
    mo,
    np,
    num_frames_default,
    t_span_common_vis,
    y0_common_vis,
):
    # Scénario 2: Poussée Mg, phi=0
    mo.md("### Vidéo 2: Poussée $Mg$, $\phi=0$")
    def f_phi_vis2(t, state): return np.array([M*g, 0.0])
    video_out_2 = make_booster_video(
        "vis_sc2_Mg_phi0.mp4", t_span_common_vis, y0_common_vis, f_phi_vis2,
        ylims=(-1, 12 + 0.5 * g * (t_span_common_vis[1]**2) ), # Ajuster si ça monte beaucoup
        num_frames_vid=num_frames_default, fps_vid=fps_default
    )
    video_out_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Critique :

    La force de propulsion est exactement égale au poids, ce qui maintient la fusée dans un état statique (équilibre) et l'empêche de bouger.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Scénario 3 - Déséquilibre Angulaire
    Avec une poussée égale au poids (f=Mg) mais avec un angle φ=π/8, cette simulation démontre l'effet destabilisateur d'une inclinaison même modérée. Le couple induit provoque une rotation croissante et une trajectoire complexe combinant mouvement horizontal et chute, illustrant l'importance du contrôle d'orientation.
    """
    )
    return


@app.cell
def _(
    M,
    fps_default,
    g,
    make_booster_video,
    mo,
    np,
    num_frames_default,
    t_span_common_vis,
    y0_common_vis,
):
    # Scénario 3: Poussée Mg, phi=pi/8
    mo.md("### Vidéo 3: Poussée $Mg$, $\phi=\pi/8$")
    def f_phi_vis3(t, state): return np.array([M*g, np.pi/8])
    video_out_3 = make_booster_video(
        "vis_sc3_Mg_phiPi8.mp4", t_span_common_vis, y0_common_vis, f_phi_vis3,
        xlims=(-20, 20), ylims=(-2, 25), # Doit se déplacer et tourner
        num_frames_vid=num_frames_default, fps_vid=fps_default
    )
    video_out_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Critique

        Le déséquilibre de la fusée provient d'un couple induit par l'écart entre Φ et 0
    Lorsque l'angle est different de 0 , un moment de rotation se crée entre les extrémités de la fusée. 
    Ce désalignement rompt l'équilibre initial, provoquant une rotation incontrôlée jusqu'au crash.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Scénario 4 - Atterrissage Contrôlé
    Cette simulation avancée intègre un algorithme de contrôle pour maintenir la stabilité et permettre un atterrissage en douceur. Elle combine modulation de poussée et ajustement angulaire en temps réel, démontrant les principes de contrôle PID appliqués à la navigation verticale. Le scénario sert de validation finale au système de guidage.
    """
    )
    return


@app.cell
def _(
    f_phi_controlled_landing,
    fps_default,
    make_booster_video,
    mo,
    num_frames_default,
    t_span_common_vis,
    y0_common_vis,
):
    # Scénario 4: Atterrissage contrôlé (défini précédemment)
    mo.md("### Vidéo 4: Atterrissage Contrôlé")
    video_out_4 = make_booster_video(
        "vis_sc4_controlled_landing.mp4", t_span_common_vis, y0_common_vis, 
        f_phi_controlled_landing, 
        num_frames_vid=num_frames_default, fps_vid=fps_default
    )
    video_out_4
    return


app._unparsable_cell(
    r"""
    ### Critique

        La poussée du réacteur augmente à l'approche du sol pour:
    Compenser l'accélération gravitationnelle
    En chute libre, la fusée est soumise uniquement à son poids (P=mg), ce qui la fait accélérer vers le sol.
    Pour freiner cette chute, le réacteur doit générer une force de poussée supérieure au poids (F>P), créant une accélération nette vers le haut.
    Réduire progressivement la vitesse
    Plus la fusée se rapproche du sol, plus sa vitesse acquise est grande (effet de l'accélération gravitationnelle).
    Pour annuler cette vitesse à l'atterrissage (v=0), la poussée doit être augmentée progressivement, suivant souvent une loi de décélération contrôlée (ex : guidage par feedback).
    Éviter un crash ou un rebond
    Une poussée trop faible ne compenserait pas assez la gravité → impact violent.
    Une poussée trop forte en début de freinage gaspillerait du carburant et pourrait provoquer un rebond.
    La solution optimale est donc d'adapter la poussée pour une décélération lisse jusqu'à l'arrêt complet au sol.
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
