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


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
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
    return FFMpegWriter, FuncAnimation, la, mpl, np, plt, sci, scipy, tqdm


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
    return (R,)


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
            # Clear the canvas and redraw everything at each step
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
    mo.show_code(mo.video(src=_filename))
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


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
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
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
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


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


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
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
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


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
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
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \begin{bmatrix}
    x \\
    \dot{x} \\
    y \\
    \dot{y} \\
    \theta \\
    \dot{\theta} \\
    f \\
    \phi
    \end{bmatrix}
    =
    \begin{bmatrix}
    ? \\
    0 \\
    ? \\
    0 \\
    0 \\
    0 \\
    M g \\
    0
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M (d/dt)^2 \Delta x &= - Mg (\Delta \theta + \Delta \phi)  \\
    M (d/dt)^2 \Delta y &= \Delta f \\
    J (d/dt)^2 \Delta \theta &= - (Mg \ell) \Delta \phi \\
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    A = 
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0  & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0  & 0 \\
    0 & 0 & 0 & 0 & 0  & 0 \\
    0 & 0 & 0 & 0 & 0  & 1 \\
    0 & 0 & 0 & 0 & 0  & 0 
    \end{bmatrix}
    \;\;\;
    B = 
    \begin{bmatrix}
    0 & 0\\ 
    0 & -g\\ 
    0 & 0\\ 
    1/M & 0\\
    0 & 0 \\
    0 & -M g \ell/J\\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(g, np):
    A = np.zeros((6, 6))
    A[0, 1] = 1.0
    A[1, 4] = -g
    A[2, 3] = 1.0
    A[4, -1] = 1.0
    A
    return (A,)


@app.cell(hide_code=True)
def _(J, M, g, l, np):
    B = np.zeros((6, 2))
    B[ 1, 1]  = -g 
    B[ 3, 0]  = 1/M
    B[-1, 1] = -M*g*l/J
    B
    return (B,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(A, la):
    # No since 0 is the only eigenvalue of A
    eigenvalues, eigenvectors = la.eig(A)
    eigenvalues
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell(hide_code=True)
def _(A, B, np):
    # Controllability
    cs = np.column_stack
    mp = np.linalg.matrix_power
    KC = cs([mp(A, k) @ B for k in range(6)])
    KC
    return (KC,)


@app.cell(hide_code=True)
def _(KC, np):
    # Yes!
    np.linalg.matrix_rank(KC) == 6
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell
def _(J, M, g, l, np):
    A_lat = np.array([
        [0, 1, 0, 0], 
        [0, 0, -g, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 0]], dtype=np.float64)
    B_lat = np.array([[0, -g, 0, - M * g * l / J]]).T

    A_lat, B_lat
    return A_lat, B_lat


@app.cell(hide_code=True)
def _(A_lat, B_lat, np):
    # Controllability
    _cs = np.column_stack
    _mp = np.linalg.matrix_power
    KC_lat = _cs([_mp(A_lat, k) @ B_lat for k in range(6)])
    KC_lat
    return (KC_lat,)


@app.cell(hide_code=True)
def _(KC_lat, np):
    np.linalg.matrix_rank(KC_lat) == 4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np):
    def make_fun_lat(phi):
        def fun_lat(t, state):
            x, dx, theta, dtheta = state
            phi_ = phi(t, state)
            #if linearized:
            d2x = -g * (theta + phi_)
            d2theta = -M * g * l / J * phi_
            #else:
            #d2x = -g * np.sin(theta + phi_)
            #d2theta = -M * g * l / J * np.sin(phi_)
            return np.array([dx, d2x, dtheta, d2theta])

        return fun_lat
    return (make_fun_lat,)


@app.cell(hide_code=True)
def _(make_fun_lat, mo, np, plt, sci):
    def lin_sim_1():
        def _phi(t, state):
            return 0.0
        _f_lat = make_fun_lat(_phi)
        _t_span = [0, 10]
        state_0 = [0, 0, 45 * np.pi/180.0, 0]
        _r = sci.solve_ivp(fun=_f_lat, y0=state_0, t_span=_t_span, dense_output=True)
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _sol_t = _r.sol(_t)
        _fig, (_ax1, _ax2) = plt.subplots(2, 1, sharex=True)
        _ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend()
        _ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.grid(True)
        _ax2.set_xlabel(r"time $t$")
        _ax2.legend()
        return mo.center(_fig)
    lin_sim_1()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(A_lat, B_lat, make_fun_lat, mo, np, plt, sci):

    def lin_sim_2():
        # Manual tuning of K (Angle only)

        K = np.array([0.0, 0.0, -1.0, -1.0])

        print("eigenvalues:", np.linalg.eig(A_lat - B_lat.reshape((-1,1)) @ K.reshape((1, -1))).eigenvalues)

        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - K.dot(state)

        #_f_lat = make_fun_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) # , linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_2()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(A_lat, B_lat, make_fun_lat, mo, np, plt, sci, scipy):
    Kpp = scipy.signal.place_poles(
        A=A_lat, 
        B=B_lat, 
        poles=1.0*np.array([-0.5, -0.51, -0.52, -0.53])
    ).gain_matrix.squeeze()

    def lin_sim_3():
        print(f"Kpp = {Kpp}")

        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - Kpp.dot(state)

        #_f_lat = make_f_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) # , linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_3()
    return (Kpp,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(A_lat, B_lat, l, make_fun_lat, mo, np, plt, sci, scipy):
    _Q = np.zeros((4,4))
    _Q[0, 0] = 1.0
    _Q[1, 1] = 0.0
    _Q[2, 2] = (2*l)**2
    _Q[3, 3] = 0.0
    _R = 10*(2*l)**2 * np.eye(1)

    _Pi = scipy.linalg.solve_continuous_are(
        a=A_lat, 
        b=B_lat, 
        q=_Q, 
        r=_R
    )
    Koc = (np.linalg.inv(_R) @ B_lat.T @ _Pi).squeeze()
    print(f"Koc = {Koc}")

    def lin_sim_4():    
        _t_span = [0, 20.0]
        _t = np.linspace(_t_span[0], _t_span[1], 1000)
        _state_0 = [0, 0, 45 * np.pi/180.0, 0]
        def _phi(t, state):
            return - Koc.dot(state)

        #_f_lat = make_fun_lat(_phi, linearized=False)
        #_r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        #_sol_t = _r.sol(_t)

        _f_lat = make_fun_lat(_phi) #, linearized=True)
        _r = sci.solve_ivp(fun=_f_lat, y0=_state_0, t_span=_t_span, dense_output=True)
        _sol_lin_t = _r.sol(_t)

        _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        _ax1.plot(_t, _sol_lin_t[0], label=r"$x(t)$ (lin.)")
        #_ax1.plot(_t, _sol_t[0], label=r"$x(t)$")
        _ax1.grid(True)
        _ax1.legend(loc="lower right")
        _ax2.plot(_t, _sol_lin_t[2], label=r"$\theta(t)$ (lin.)")
        #_ax2.plot(_t, _sol_t[2], label=r"$\theta(t)$")
        _ax2.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax2.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax2.grid(True)
        _ax2.legend(loc="lower right")
        _ax3.plot(_t, _phi(_t, _sol_lin_t), label=r"$\phi(t)$ (lin.)")
        #_ax3.plot(_t, _phi(_t, _sol_t), label=r"$\phi(t)$")
        _ax3.grid(True)
        _ax3.plot(_t, 0.5 * np.pi * np.ones_like(_t), "r--", label=r"$\pm\pi/2$")
        _ax3.plot(_t, -0.5 * np.pi * np.ones_like(_t), "r--")
        _ax3.set_xlabel(r"time $t$")
        _ax3.legend(loc="lower right")
        return mo.center(_fig)

    lin_sim_4()
    return (Koc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    Kpp,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_Kpp():
        t_span = [0.0, 20.0]
        y0 = [0.0, 0.0, 20.0, 0.0, 45 * np.pi/180.0, 0.0]
        def f_phi(t, state):
            x, dx, y, dy, theta, dtheta = state  
            return np.array(
                [M*g, -Kpp.dot([x, dx, theta, dtheta])]
            )
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_Kpp.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +24*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_Kpp())

    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    Koc,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_Koc():
        t_span = [0.0, 20.0]
        y0 = [0.0, 0.0, 20.0, 0.0, 45 * np.pi/180.0, 0.0]
        def f_phi(t, state):
            x, dx, y, dy, theta, dtheta = state  
            return np.array(
                [M*g, -Koc.dot([x, dx, theta, dtheta])]
            )
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_Koc.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +24*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_Koc())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Exact Linearization


    Consider an auxiliary system which is meant to compute the force $(f_x, f_y)$ applied to the booster. 

    Its inputs are 

    $$
    v = (v_1, v_2) \in \mathbb{R}^2,
    $$

    its dynamics 

    $$
    \ddot{z} = v_1 \qquad \text{ where } \quad z\in \mathbb{R}
    $$ 

    and its output $(f_x, f_y) \in \mathbb{R}^2$ is given by

    \[
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} = R\left(\theta - \frac{\pi}{2}\right)
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    \]

    ⚠️ Note that the second component $f_y$ of the reactor force is undefined whenever $z=0$.

    Consider the output $h$ of the original system

    $$
    h := 
    \begin{bmatrix}
    x - (\ell/3) \sin \theta \\
    y + (\ell/3) \cos \theta
    \end{bmatrix} \in \mathbb{R}^2
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Geometrical Interpretation

    Provide a geometrical interpretation of $h$ (for example, make a drawing).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""🔓 The coordinates $h$ represent a fixed point on the booster. Start from the reactor, move to the center of mass (distance $\ell$) then continue for $\ell/3$ in this direction. The coordinates of this point are $h$.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 First and Second-Order Derivatives

    Compute $\dot{h}$ as a function of $\dot{x}$, $\dot{y}$, $\theta$ and $\dot{\theta}$ (and constants) and then $\ddot{h}$ as a function of $\theta$ and $z$ (and constants) when the auxiliary system is plugged in the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 We have 

    $$
    \boxed{
    \dot{h} = 
    \begin{bmatrix}
    \dot{x} - (\ell /3)  (\cos \theta) \dot{\theta} \\
    \dot{y} - (\ell /3) (\sin \theta) \dot{\theta}
    \end{bmatrix}}
    $$

    and therefore

    \begin{align*}
    \ddot{h} &=
    \begin{bmatrix}
    \ddot{x} - (\ell/3)\cos\theta\, \ddot{\theta} + (\ell/3)\sin\theta\, \dot{\theta}^2 \\
    \ddot{y} - (\ell/3)\sin\theta\, \ddot{\theta} - (\ell/3)\cos\theta\, \dot{\theta}^2
    \end{bmatrix} \\
    &=
    \begin{bmatrix}
    \frac{f_x}{M} - \frac{\ell}{3} \cos\theta \cdot \frac{3}{M\ell} (\cos\theta\, f_x + \sin\theta\, f_y) + \frac{\ell}{3} \sin\theta\, \dot{\theta}^2 \\
    \frac{f_y}{M} - g - \frac{\ell}{3} \sin\theta \cdot \frac{3}{M\ell} (\cos\theta\, f_x + \sin\theta\, f_y) - \frac{\ell}{3} \cos\theta\, \dot{\theta}^2
    \end{bmatrix} \\
    &=
    \frac{1}{M}
    \begin{bmatrix}
    \sin\theta \\
    -\cos\theta
    \end{bmatrix}
    \left(
    \begin{bmatrix}
    \sin\theta & -\cos\theta
    \end{bmatrix}
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix}
    + M\frac{\ell}{3} \dot{\theta}^2
    \right)
    -
    \begin{bmatrix}
    0 \\
    g
    \end{bmatrix}
    \end{align*}


    On the other hand, since

    \[
    \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} = R\left(\theta - \frac{\pi}{2}\right)
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    \]

    we have

    $$
    \begin{bmatrix}
    z - M\frac{\ell}{3}\dot{\theta}^2 \\
    \frac{M\ell v_2}{3z}
    \end{bmatrix}
    = R\left(\frac{\pi}{2} - \theta\right) \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} =
    \begin{bmatrix}
    \sin \theta & - \cos \theta \\
    \cos \theta & \sin \theta
    \end{bmatrix}
    \begin{bmatrix}
    f_x \\ f_y
    \end{bmatrix}
    $$

    and therefore we end up with

    $$
    \boxed{\ddot{h} = 
      \frac{1}{M}
      \begin{bmatrix}
        \sin\theta \\
        -\cos\theta
       \end{bmatrix}
      z
      -
      \begin{bmatrix}
        0 \\
        g
      \end{bmatrix}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Third and Fourth-Order Derivatives 

    Compute the third derivative $h^{(3)}$ of $h$ as a function of $\theta$ and $z$ (and constants) and then the fourth derivative $h^{(4)}$ of $h$ with respect to time as a function of $\theta$, $\dot{\theta}$, $z$, $\dot{z}$, $v$ (and constants) when the auxiliary system is on.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 We have 

    \[
    \boxed{
    h^{(3)} = \frac{1}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}z + \frac{1}{M}
    \begin{bmatrix}
    \sin \theta \\
    -\cos \theta
    \end{bmatrix}
    \dot{z}
    }
    \]

    and consequently

    \[
    \begin{aligned}
    h^{(4)} &= \frac{1}{M}
    \begin{bmatrix}
    -\sin \theta \\
    \cos \theta
    \end{bmatrix}
    \dot{\theta}^2 z + \frac{1}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \frac{3}{Ml} (\cos \theta f_x + \sin \theta f_y) z \\
    &+ \frac{2}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}\dot{z} + \frac{1}{M}
    \begin{bmatrix}
    \sin \theta \\
    -\cos \theta
    \end{bmatrix}
    v_1
    \end{aligned}
    \]

    Since

    \[
    \begin{bmatrix}
    z - \frac{Ml}{3} \dot{\theta}^2 \\
    \frac{Mlv_2}{3z}
    \end{bmatrix}
    = R\left(\frac{\pi}{2} - \theta\right) \begin{bmatrix}
    f_x \\
    f_y
    \end{bmatrix} =
    \begin{bmatrix}
    \sin \theta f_x - \cos \theta f_y \\
    \cos \theta f_x + \sin \theta f_y
    \end{bmatrix}
    \]

    we have

    \[
    h^{(4)} = \frac{1}{M}
    \begin{bmatrix}
    \sin \theta & \cos \theta \\
    -\cos \theta & \sin \theta
    \end{bmatrix}
    \begin{bmatrix}
    v_1 \\
    v_2
    \end{bmatrix}
    + \frac{1}{M}
    \begin{bmatrix}
    -\sin \theta \\
    \cos \theta
    \end{bmatrix}
    \dot{\theta}^2 z
    + \frac{2}{M}
    \begin{bmatrix}
    \cos \theta \\
    \sin \theta
    \end{bmatrix}
    \dot{\theta}\dot{z}
    \]

    \[
    \boxed{
    h^{(4)}
    = \frac{1}{M} R \left( \theta - \frac{\pi}{2} \right)
    \left(
    v +
    \begin{bmatrix}
    -\dot{\theta}^2 z \\
    2 \dot{\theta} \dot{z}
    \end{bmatrix}
    \right)
    }
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Exact Linearization

    Show that with yet another auxiliary system with input $u=(u_1, u_2)$ and output $v$ fed into the previous one, we can achieve the dynamics

    $$
    h^{(4)} = u
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    🔓 Since

    \[
    h^{(4)}
    = \frac{1}{M} R \left( \theta - \frac{\pi}{2} \right)
    \left(
    v +
    \begin{bmatrix}
    -\dot{\theta}^2 z \\
    2 \dot{\theta} \dot{z}
    \end{bmatrix}
    \right)
    \]  

    we can define $v$ as 

    $$
    \boxed{
    v =
    M \, R \left(\frac{\pi}{2} - \theta \right)
    u + 
    \begin{bmatrix}
    \dot{\theta}^2 z \\
    -2 \dot{\theta} \dot{z}
    \end{bmatrix}
    }
    $$

    and achieve the desired result.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 State to Derivatives of the Output

    Implement a function `T` of `x, dx, y, dy, theta, dtheta, z, dz` that returns `h_x, h_y, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y`.
    """
    )
    return


@app.cell
def _(M, g, l, np):
    def T(x, dx, y, dy, theta, dtheta, z, dz):

        h_x = x - (l / 3) * np.sin(theta)
        h_y = y + (l / 3) * np.cos(theta)

        dh_x = dx - (l / 3) * np.cos(theta) * dtheta
        dh_y = dy - (l / 3) * np.sin(theta) * dtheta

        d2h_x = (z / M) * np.sin(theta)
        d2h_y = (z / M) * (-np.cos(theta)) - g

        d3h_x = (z * dtheta / M) * np.cos(theta) + (dz / M) * np.sin(theta)
        d3h_y = (z * dtheta / M) * np.sin(theta) - (dz / M) * np.cos(theta)

        return h_x, h_y, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y


    return (T,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Inversion 


    Assume for the sake of simplicity that $z<0$ at all times. Show that given the values of $h$, $\dot{h}$, $\ddot{h}$ and $h^{(3)}$, one can uniquely compute the booster state (the values of $x$, $\dot{x}$, $y$, $\dot{y}$, $\theta$, $\dot{\theta}$) and auxiliary system state (the values of $z$ and $\dot{z}$).

    Implement the corresponding function `T_inv`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Étant donné les relations suivantes entre les dérivées de la sortie \(h\) et l’état \(\bigl[x,\dot x, y,\dot y,\theta,\dot\theta,z,\dot z\bigr]\) :

    1.  \(h_x = x - \tfrac{\ell}{3}\sin\theta\)  
    2.  \(h_y = y + \tfrac{\ell}{3}\cos\theta\)  
    3.  \(\dot h_x = \dot x - \tfrac{\ell}{3}\,\dot\theta\cos\theta\)  
    4.  \(\dot h_y = \dot y - \tfrac{\ell}{3}\,\dot\theta\sin\theta\)  
    5.  \(\ddot h_x = \tfrac{z}{M}\,\sin\theta\)  
    6.  \(\ddot h_y = -\,\tfrac{z}{M}\,\cos\theta \;-\; g\)  
    7.  \(\dddot h_x = \tfrac{z\,\dot\theta}{M}\cos\theta \;+\; \tfrac{\dot z}{M}\sin\theta\)  
    8.  \(\dddot h_y = \tfrac{z\,\dot\theta}{M}\sin\theta \;-\; \tfrac{\dot z}{M}\cos\theta\)  

    On suppose \(z<0\). Montrons comment remonter de \(\{h,\dot h,\ddot h,\dddot h\}\) à l’état complet.

    ---

    #### Étape 1 : extraire \(z\) et \(\theta\) de \(\ddot h\)

    Définissons  

    \[
    V_x = \ddot h_x,\quad
    V_y = \ddot h_y + g.
    \]

    Alors, d’après (5) et (6) :

    \[
    M\,V_x = z\sin\theta,
    \quad
    M\,V_y = -\,z\cos\theta.
    \]

    En sommant les carrés :

    \[
    (MV_x)^2 + (MV_y)^2 
    = z^2(\sin^2\theta + \cos^2\theta)
    = z^2,
    \]

    d’où, comme \(z<0\),

    \[
    \boxed{z = -\,M\,\sqrt{V_x^2 + V_y^2}
    = -\,M\,\sqrt{\ddot h_x^2 + (\ddot h_y + g)^2}.}
    \]

    Puis :

    \[
    \sin\theta = \frac{M\,V_x}{z},
    \quad
    \cos\theta = -\,\frac{M\,V_y}{z},
    \]

    et donc

    (la fonction atan2 à deux arguments est une variante de la fonction arctan classique utile pour ce cas.)

    \[
    \boxed{\theta = \operatorname{atan2}\!\bigl(\sin\theta,\cos\theta\bigr)
    = \operatorname{atan2}\!\bigl(\ddot h_x, -(\ddot h_y + g)\bigr).}
    \]


    ---

    #### Étape 2 : extraire \(\dot z\) et \(\dot\theta\) de \(\dddot h\)

    D’après (7) et (8) :

    \[
    \begin{cases}
    M\,\dddot h_x = z\,\dot\theta\cos\theta + \dot z\,\sin\theta,\\[4pt]
    M\,\dddot h_y = z\,\dot\theta\sin\theta - \dot z\,\cos\theta.
    \end{cases}
    \]

    — Multiplier la première par \(\sin\theta\) et la seconde par \(\cos\theta\), puis **soustraire** la seconde de la première :

    \[
    \dot z
    = M\bigl(\dddot h_x\,\sin\theta - \dddot h_y\,\cos\theta\bigr).
    \]

    — Multiplier la première par \(\cos\theta\) et la seconde par \(\sin\theta\), puis **additionner** :

    \[
    z\,\dot\theta
    = M\bigl(\dddot h_x\,\cos\theta + \dddot h_y\,\sin\theta\bigr),
    \quad
    \dot\theta = \frac{M}{z}\bigl(\dddot h_x\,\cos\theta + \dddot h_y\,\sin\theta\bigr).
    \]

    ---

    #### Étape 3 : extraire \(\dot x\) et \(\dot y\) de \(\dot h\)

    D’après (3) et (4) :

    \[
    \boxed{\dot x = \dot h_x + \tfrac{\ell}{3}\,\dot\theta\cos\theta,\quad
    \dot y = \dot h_y + \tfrac{\ell}{3}\,\dot\theta\sin\theta.}
    \]

    ---

    #### Étape 4 : extraire \(x\) et \(y\) de \(h\)

    D’après (1) et (2) :

    \[
    \boxed{x = h_x + \tfrac{\ell}{3}\,\sin\theta,\quad
    y = h_y - \tfrac{\ell}{3}\,\cos\theta.}
    \]

    ---

    **Conclusion :** sous l’hypothèse \(z<0\), les dérivées  
    \(\{h,\dot h,\ddot h,\dddot h\}\) déterminent de façon unique  
    \(\{x,\dot x,y,\dot y,\theta,\dot\theta,z,\dot z\}\).
    """
    )
    return


@app.cell
def _(M, g, l, np):
    def T_inv(h_x, h_y, dh_x, dh_y, d2h_x, d2h_y, d3h_x, d3h_y):
        Vx = M * d2h_x
        Vy = M * (d2h_y + g)
        z_squared = Vx**2 + Vy**2
        if z_squared < 1e-9:

             raise ValueError("Computed z is non-negative or near zero, violating the assumption z < 0.")
        else:
            z = -np.sqrt(z_squared)
        theta = np.arctan2(Vy, Vx) - np.pi / 2.0
        theta = np.arctan2(-d2h_x, d2h_y + g)

        dz = M * (d3h_x * np.sin(theta) - d3h_y * np.cos(theta))

        if np.abs(z) < 1e-9:
            dtheta = 0.0
        else:
            dtheta = M * (d3h_x * np.cos(theta) + d3h_y * np.sin(theta)) / z

        x = h_x + (l / 3.0) * np.sin(theta)
        y = h_y - (l / 3.0) * np.cos(theta)
        dx = dh_x + (l / 3.0) * np.cos(theta) * dtheta
        dy = dh_y + (l / 3.0) * np.sin(theta) * dtheta
        return x, dx, y, dy, theta, dtheta, z, dz

    return (T_inv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Admissible Path Computation

    Implement a function

    ```python
    def compute(
        x_0,
        dx_0,
        y_0,
        dy_0,
        theta_0,
        dtheta_0,
        z_0,
        dz_0,
        x_tf,
        dx_tf,
        y_tf,
        dy_tf,
        theta_tf,
        dtheta_tf,
        z_tf,
        dz_tf,
        tf,
    ):
        ...

    ```

    that returns a function `fun` such that `fun(t)` is a value of `x, dx, y, dy, theta, dtheta, z, dz, f, phi` at time `t` that match the initial and final values provided as arguments to `compute`.
    """
    )
    return


@app.cell
def _(M, R, T, T_inv, l, np):
    def poly7_coeffs(val0, dval0, d2val0, d3val0, valtf, dvaltf, d2valtf, d3valtf, tf):

        a0 = val0
        a1 = dval0
        a2 = d2val0 / 2.0
        a3 = d3val0 / 6.0

        rhs = np.zeros(4)
        rhs[0] = valtf - (a0 + a1*tf + a2*tf**2 + a3*tf**3)
        rhs[1] = dvaltf - (a1 + 2*a2*tf + 3*a3*tf**2)
        rhs[2] = d2valtf - (2*a2 + 6*a3*tf)
        rhs[3] = d3valtf - (6*a3)

        matrix = np.zeros((4, 4))
        matrix[0, :] = [tf**4, tf**5, tf**6, tf**7]
        matrix[1, :] = [4*tf**3, 5*tf**4, 6*tf**5, 7*tf**6]
        matrix[2, :] = [12*tf**2, 20*tf**3, 30*tf**4, 42*tf**5]
        matrix[3, :] = [24*tf**1, 60*tf**2, 120*tf**3, 210*tf**4]

        try:
            a4, a5, a6, a7 = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as e:
            print(f"Error solving polynomial coefficients: {e}")
            print(f"Matrix:\n{matrix}")
            print(f"RHS:\n{rhs}")
            raise e 

        return np.array([a0, a1, a2, a3, a4, a5, a6, a7])

    def poly_eval(coeffs, t, derivative_order=0):
        p_coeffs = np.copy(coeffs)
        for k in range(derivative_order):
            if k + 1 > len(p_coeffs) - derivative_order + k:

                 return 0.0 

            for i in range(k + 1, len(p_coeffs)):
                p_coeffs[i] = p_coeffs[i] * (i - k)

        val = 0.0
        t_power = 1.0

        val = 0.0
        for i in range(derivative_order, len(coeffs)):
            coeff = coeffs[i]
            factorial_ratio = 1.0
            for k in range(i, i - derivative_order, -1):
                 factorial_ratio *= k

            val += coeff * factorial_ratio * (t**(i - derivative_order))

        return val
    def compute(
        x_0,
        dx_0,
        y_0,
        dy_0,
        theta_0,
        dtheta_0,
        z_0,
        dz_0,
        x_tf,
        dx_tf,
        y_tf,
        dy_tf,
        theta_tf,
        dtheta_tf,
        z_tf,
        dz_tf,
        tf,
    ):

        if tf <= 0:
            raise ValueError("Final time tf must be positive.")

        h_x0, h_y0, dh_x0, dh_y0, d2h_x0, d2h_y0, d3h_x0, d3h_y0 = T(
            x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0
        )
        h_xtf, h_ytf, dh_xtf, dh_ytf, d2h_xtf, d2h_y_tf, d3h_xtf, d3h_y_tf = T(
            x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf
        )

        coeffs_hx = poly7_coeffs(h_x0, dh_x0, d2h_x0, d3h_x0, h_xtf, dh_xtf, d2h_xtf, d3h_xtf, tf)
        coeffs_hy = poly7_coeffs(h_y0, dh_y0, d2h_y_tf, d3h_y_tf, h_ytf, dh_ytf, d2h_y_tf, d3h_y_tf, tf)


        def fun(t):

            if t < 0 or t > tf:

                 return None 


            h_x_t = poly_eval(coeffs_hx, t, 0)
            h_y_t = poly_eval(coeffs_hy, t, 0)
            dh_x_t = poly_eval(coeffs_hx, t, 1)
            dh_y_t = poly_eval(coeffs_hy, t, 1)
            d2h_x_t = poly_eval(coeffs_hx, t, 2)
            d2h_y_t = poly_eval(coeffs_hy, t, 2)
            d3h_x_t = poly_eval(coeffs_hx, t, 3)
            d3h_y_t = poly_eval(coeffs_hy, t, 3)
            d4h_x_t = poly_eval(coeffs_hx, t, 4) # This is u1
            d4h_y_t = poly_eval(coeffs_hy, t, 4) # This is u2
            u = np.array([d4h_x_t, d4h_y_t])


            try:
                x_t, dx_t, y_t, dy_t, theta_t, dtheta_t, z_t, dz_t = T_inv(
                    h_x_t, h_y_t, dh_x_t, dh_y_t, d2h_x_t, d2h_y_t, d3h_x_t, d3h_y_t
                )
            except ValueError as e:

                print(f"T_inv failed at t={t:.4f}: {e}")
                return None

            R_pi2_minus_theta = R(np.pi/2.0 - theta_t)
            v_vec = M * R_pi2_minus_theta @ u + np.array([dtheta_t**2 * z_t, -2 * dtheta_t * dz_t])
            v1_t, v2_t = v_vec


            z_prime = z_t - M * (l / 3.0) * dtheta_t**2

            if np.abs(z_t) < 1e-9:

                 print(f"Auxiliary state z is near zero at t={t:.4f} during input calculation.")
                 return None 

            z_perp_term = (M * l * v2_t) / (3.0 * z_t)

            R_theta_minus_pi2 = R(theta_t - np.pi/2.0)
            f_vec = R_theta_minus_pi2 @ np.array([z_prime, z_perp_term])
            fx_t, fy_t = f_vec

            f_t = np.sqrt(fx_t**2 + fy_t**2)


            phi_t = 0.0 
            if f_t > 1e-9: 

                phi_t = np.arctan2(fx_t, fy_t) - theta_t

            return (x_t, dx_t, y_t, dy_t, theta_t, dtheta_t, z_t, dz_t, f_t, phi_t)


        return fun
    return compute, poly_eval


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Graphical Validation

    Test your `compute` function with

      - `x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0 = 5.0, 0.0, 20.0, -1.0, -np.pi/8, 0.0, -M*g, 0.0`,
      - `x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf = 0.0, 0.0, 4/3*l, 0.0,     0.0, 0.0, -M*g, 0.0`,
      - `tf = 10.0`.

    Make the graph of the relevant variables as a function of time, then make a video out of the same result. Comment and iterate if necessary!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remarque :

    Le résultat affiché correspond à une planification de trajectoire initiale qui fournit une solution réalisable des le debut.
    """
    )
    return


@app.cell
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    T,
    T_inv,
    coeffs_hx,
    coeffs_hy,
    compute,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    poly_eval,
    tqdm,
):
    x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0 = 5.0, 0.0, 20.0, -1.0, -np.pi/8, 0.0, -M*g, 0.0

    x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf = 0.0, 0.0, 4/3.0 * l, 0.0, 0.0, 0.0, -M*g, 0.0

    tf = 20.0

    try:
        trajectory_fun = compute(
            x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0,
            x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf,
            tf
        )

        n_points = 400
        t_points = np.linspace(0, tf, n_points)

        print("Evaluating trajectory function...")
        trajectory_data_list = [trajectory_fun(t) for t in tqdm(t_points, desc="Evaluating points")]
        print("Evaluation complete.")

        first_none_idx = next((i for i, data in enumerate(trajectory_data_list) if data is None), len(t_points))

        t_points_plot = t_points[:first_none_idx]
        trajectory_data_plot = np.array([trajectory_data_list[i] for i in range(first_none_idx)])

        print(f"Shape of successfully computed trajectory data: {trajectory_data_plot.shape}")

        if trajectory_data_plot.shape[0] > 1:

            print("Generating plots...")

            x_t, dx_t, y_t, dy_t, theta_t, dtheta_t, z_t, dz_t, f_t, phi_t = trajectory_data_plot.T

            fig_traj, axes_traj = plt.subplots(5, 1, sharex=True, figsize=(10, 12))

            axes_traj[0].plot(t_points_plot, x_t, label=r"$x(t)$")
            axes_traj[0].plot(t_points_plot, y_t, label=r"$y(t)$")
            axes_traj[0].axhline(y_tf, color='grey', linestyle='--', label=r"$y_{tf}$")
            axes_traj[0].set_ylabel("Position (m)")
            axes_traj[0].grid(True)
            axes_traj[0].legend()

            axes_traj[1].plot(t_points_plot, dx_t, label=r"$\dot{x}(t)$")
            axes_traj[1].plot(t_points_plot, dy_t, label=r"$\dot{y}(t)$")
            axes_traj[1].set_ylabel("Velocity (m/s)")
            axes_traj[1].grid(True)
            axes_traj[1].legend()

            axes_traj[2].plot(t_points_plot, theta_t * 180/np.pi, label=r"$\theta(t)$")
            axes_traj[2].axhline(theta_tf * 180/np.pi, color='grey', linestyle='--', label=r"$\theta_{tf}$")
            axes_traj[2].set_ylabel("Tilt (deg)")
            axes_traj[2].grid(True)
            axes_traj[2].legend()

            axes_traj[3].plot(t_points_plot, dtheta_t * 180/np.pi, label=r"$\dot{\theta}(t)$")
            axes_traj[3].set_ylabel("Angular Vel (deg/s)")
            axes_traj[3].grid(True)
            axes_traj[3].legend()

            axes_traj[4].plot(t_points_plot, f_t, label=r"$f(t)$")
            axes_traj[4].plot(t_points_plot, phi_t * 180/np.pi, label=r"$\phi(t)$ (deg)")
            axes_traj[4].axhline(M*g, color='red', linestyle='--', label=r"$Mg$ (Equil. Force)")
            axes_traj[4].axhline(90, color='orange', linestyle='--', label=r"$\pm \pi/2$ (90$^\circ$)")
            axes_traj[4].axhline(-90, color='orange', linestyle='--')
            axes_traj[4].set_ylabel("Input")
            axes_traj[4].set_xlabel("time $t$ (s)")
            axes_traj[4].grid(True)
            axes_traj[4].legend()

            plt.suptitle("Exact Linearization Trajectory")
            plt.tight_layout()

            mo.center(fig_traj)
            print("Plots displayed.")

            print("--- Boundary Condition Check ---")
            print("Initial state (computed):", np.round(trajectory_data_plot[0, :8], 6))
            print("Initial state (desired): ", np.round(np.array([x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0]), 6))

            if first_none_idx == len(t_points):
                 print("Final state (computed):  ", np.round(trajectory_data_plot[-1, :8], 6))
                 print("Final state (desired):   ", np.round(np.array([x_tf, dx_tf, y_tf, dy_tf, theta_tf, dtheta_tf, z_tf, dz_tf]), 6))
            else:
                 print(f"Trajectory computation stopped at t = {t_points_plot[-1]:.4f} < tf={tf} due to inadmissibility (e.g., z >= 0).")

            print(f"Min f: {np.min(f_t):.4f}")
            print(f"Max f: {np.max(f_t):.4f}")
            print(f"Min phi (deg): {np.min(phi_t) * 180/np.pi:.4f}")
            print(f"Max phi (deg): {np.max(phi_t) * 180/np.pi:.4f}")
            print(f"Min z: {np.min(z_t):.4f}")
            print(f"Max z: {np.max(z_t):.4f}")

            video_indices = list(range(trajectory_data_plot.shape[0]))

            if len(video_indices) > 1:
                print("Attempting to generate video...")
                def animate_exact_linearization(i):
                    state_inputs = trajectory_data_plot[i]
                    t = t_points_plot[i]
                    x, dx, y, dy, theta, dtheta, z, dz, f, phi = state_inputs

                    axes_video.clear()
                    draw_booster(x, y, theta, f, phi, axes=axes_video)

                    max_abs_x = np.max(np.abs(x_t))
                    min_y = np.min(y_t)
                    max_y = np.max(y_t)

                    axes_video.set_xlim(-max_abs_x - 2*l, max_abs_x + 2*l)
                    axes_video.set_ylim(min(min_y, -2*l) - l, max_y + 2*l)
                    axes_video.set_aspect("equal")
                    axes_video.grid(True)
                    axes_video.set_xlabel("x (m)")
                    axes_video.set_ylabel("y (m)")
                    axes_video.set_title(f"Exact Linearization Trajectory (t={t:.2f} s)")
                    axes_video.set_axisbelow(True)

                fig_video, axes_video = plt.subplots(figsize=(8, 10))

                video_exact_linearization_path = "sim_exact_linearization.mp4"

                duration = t_points_plot[-1] - t_points_plot[0]
                fps = len(video_indices) / duration if duration > 0 and len(video_indices) > 1 else 30
                if fps > 60: fps = 60

                try:
                    print(f"Attempting to save video to {video_exact_linearization_path}...")
                    writer = FFMpegWriter(fps=fps)

                    pbar = tqdm(total=len(video_indices), desc="Generating exact linearization video")
                    def wrapper_animate_indexed(i):
                         animate_exact_linearization(i)
                         pbar.update(1)

                    anim = FuncAnimation(fig_video, wrapper_animate_indexed, frames=video_indices, repeat=False)
                    anim.save(video_exact_linearization_path, writer=writer)
                    pbar.close()
                    print("Video saving complete.")

                    import os
                    if os.path.exists(video_exact_linearization_path) and os.path.getsize(video_exact_linearization_path) > 0:
                        print(f"Displaying video from {video_exact_linearization_path}...")
                        mo.video(src=video_exact_linearization_path)
                    else:
                        print(f"Video file {video_exact_linearization_path} was not created or is empty.")
                except Exception as video_save_error:
                    print(f"An error occurred during video saving: {video_save_error}")
                finally:
                    plt.close(fig_video)
            else:
                 print("Not enough valid data points (less than 2) to generate a video.")
        else:
             print("Not enough valid data points (less than 2) to generate plots or video.")
             print(f"Computed trajectory data has shape {trajectory_data_plot.shape[0]}, which is less than 2.")
             if trajectory_data_list and trajectory_data_list[0] is not None:
                 print("Debug info from t=0:")
                 print(f"  State, Aux, Inputs at t=0: {np.round(np.array(trajectory_data_list[0]), 6)}")
                 try:
                      h0_derivs_computed = T(x_0, dx_0, y_0, dy_0, theta_0, dtheta_0, z_0, dz_0)
                      state_recovered_t0 = T_inv(*h0_derivs_computed)
                      print(f"  T_inv(T(t=0 state)) result: {np.round(np.array(state_recovered_t0), 6)}")
                      h_x0_poly = poly_eval(coeffs_hx, 0, 0); h_y0_poly = poly_eval(coeffs_hy, 0, 0)
                      dh_x0_poly = poly_eval(coeffs_hx, 0, 1); dh_y0_poly = poly_eval(coeffs_hy, 0, 1)
                      d2h_x0_poly = poly_eval(coeffs_hx, 0, 2); d2h_y0_poly = poly_eval(coeffs_hy, 0, 2)
                      d3h_x0_poly = poly_eval(coeffs_hx, 0, 3); d3h_y0_poly = poly_eval(coeffs_hy, 0, 3)
                      state_recovered_poly0 = T_inv(h_x0_poly, h_y0_poly, dh_x0_poly, dh_y0_poly, d2h_x0_poly, d2h_y0_poly, d3h_x0_poly, d3h_y0_poly)
                      print(f"  T_inv(Poly_eval(t=0)) result: {np.round(np.array(state_recovered_poly0), 6)}")
                 except Exception as debug_e:
                      print(f"  Debug check at t=0 failed: {debug_e}")
    except Exception as e:
        print(f"An error occurred during trajectory computation or plotting: {e}")

    mo.video(src=video_exact_linearization_path)

    return (fig_traj,)


@app.cell
def _(fig_traj, mo):
    mo.center(fig_traj)
    return


if __name__ == "__main__":
    app.run()
