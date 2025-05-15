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
    return FFMpegWriter, FuncAnimation, la, mpl, np, plt, scipy, tqdm


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


@app.cell(hide_code=True)
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Points d'équilibre

    Un point d'équilibre \((x_e, \dot{x}_e, y_e, \dot{y}_e, \theta_e, \dot{\theta}_e)\)  
    avec des entrées constantes \(f_e, \phi_e\) est un état où toutes les dérivées par rapport au temps sont nulles.

    $$
    \dot{x}_e = 0 \\
    \dot{y}_e = 0 \\
    \dot{\theta}_e = 0
    $$

    $$
    \ddot{x}_e = 0 \\
    \ddot{y}_e = 0 \\
    \ddot{\theta}_e = 0
    $$

    D’après les équations du mouvement :

    $$
    M \,\ddot{x}_e = -f_e \,\sin(\theta_e + \phi_e) = 0
    $$

    Comme \(f_e > 0\), il faut \(\sin(\theta_e + \phi_e)=0\), d’où

    $$
    \theta_e + \phi_e = k\pi,\quad k\in\mathbb{Z}.
    $$

    Dans les plages \(|\theta_e|<\tfrac{\pi}{2}\) et \(|\phi_e|<\tfrac{\pi}{2}\),  
    la seule solution est \(\theta_e+\phi_e=0\), donc  

    $$
    \phi_e = -\theta_e.
    $$

    ---

    $$
    M \,\ddot{y}_e = f_e\,\cos(\theta_e + \phi_e) - M g = 0
    $$

    Avec \(\theta_e+\phi_e=0\), on obtient  

    $$
    f_e\cos(0) - M g = 0
    \quad\Longrightarrow\quad
    f_e = M g.
    $$

    ---

    $$
    J \,\ddot{\theta}_e = -\ell\,f_e\,\sin(\phi_e) = 0
    $$

    Ici, \(\ell>0\) et \(f_e=Mg>0\), donc \(\sin(\phi_e)=0\), d’où  

    $$
    \phi_e = n\pi,\quad n\in\mathbb{Z}.
    $$

    Avec \(|\phi_e|<\tfrac{\pi}{2}\), la seule possibilité est  

    $$
    \phi_e = 0.
    $$

    ---

    En combinant \(\phi_e=0\) et \(\phi_e=-\theta_e\), on obtient 

    $$
    \theta_e = 0.
    $$

    **Conclusion**  
    L’unique état d’équilibre (pour \(|\theta_e|,|\phi_e|<\tfrac{\pi}{2}\)) est :

    $$
    \theta_e = 0,\quad \dot{\theta}_e = 0,\quad \dot{x}_e = 0,\quad \dot{y}_e = 0,
    $$

    avec

    $$
    f_e = M g,\quad \phi_e = 0,
    $$

    et \(x_e,y_e\) libres (typiquement choisis au point d’atterrissage désiré, par exemple \((0,\ell)\)).
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


@app.cell
def _(mo):
    mo.md(
        r"""
    Soit le vecteur d'état $z = [x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}]^T$ et le vecteur d'entrée $u = [f, \phi]^T$.
    La dynamique du système est $\dot{z} = F(z, u)$, avec l'équilibre $(z_e, u_e)$, où $z_e = [x_e, 0, y_e, 0, 0, 0]^T$ et $u_e = [Mg, 0]^T$.

    La linéarisation autour de $(z_e,u_e)$ donne $\Delta\dot{z} \approx A\,\Delta z + B\,\Delta u$, où $\Delta z = z - z_e$, $\Delta u = u - u_e$, et $A = \left.\frac{\partial F}{\partial z}\right|_{(z_e,u_e)}$, $B = \left.\frac{\partial F}{\partial u}\right|_{(z_e,u_e)}$.

    La fonction $F(z,u)$ décrivant la dynamique est :

    $$
    F(z,u) =
    \begin{bmatrix}
    \dot{x} \\[6pt]
    -\dfrac{f\,\sin(\theta+\phi)}{M} \\[8pt]
    \dot{y} \\[6pt]
    \dfrac{f\,\cos(\theta+\phi)}{M} - g \\[8pt]
    \dot{\theta} \\[6pt]
    -\dfrac{\ell\,f\,\sin\phi}{J}
    \end{bmatrix}.
    $$

    On calcule maintenant les matrices $A$ et $B$ en évaluant les dérivées partielles à l’équilibre $(z_e,u_e)$.

    $$
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    $$

    La matrice $B$ contient les dérivées partielles de $\dot{z}$ par rapport aux entrées $f$ et $\phi$, évaluées à l'équilibre $(f_e, \phi_e) = (Mg, 0)$. Le dernier terme, $\frac{\partial \ddot{\theta}}{\partial \phi}|_e$, est $\frac{\partial}{\partial \phi}\left(-\frac{\ell f \sin \phi}{J}\right) = -\frac{\ell f \cos \phi}{J}$. Évalué à l'équilibre, cela donne $-\frac{\ell Mg \cos(0)}{J} = -\frac{\ell Mg}{J}$. En substituant $J=\tfrac{M\ell^2}{3}$, ce terme devient $-\frac{\ell Mg}{M\ell^2/3} = -\frac{3g}{\ell}$.

    $$
    B =
    \begin{bmatrix}
    \left.\frac{\partial \dot{x}}{\partial f}\right|_e & \left.\frac{\partial \dot{x}}{\partial \phi}\right|_e \\
    \left.\frac{\partial \ddot{x}}{\partial f}\right|_e & \left.\frac{\partial \ddot{x}}{\partial \phi}\right|_e \\
    \left.\frac{\partial \dot{y}}{\partial f}\right|_e & \left.\frac{\partial \dot{y}}{\partial \phi}\right|_e \\
    \left.\frac{\partial \ddot{y}}{\partial f}\right|_e & \left.\frac{\partial \ddot{y}}{\partial \phi}\right|_e \\
    \left.\frac{\partial \dot{\theta}}{\partial f}\right|_e & \left.\frac{\partial \dot{\theta}}{\partial \phi}\right|_e \\
    \left.\frac{\partial \ddot{\theta}}{\partial f}\right|_e & \left.\frac{\partial \ddot{\theta}}{\partial \phi}\right|_e
    \end{bmatrix} =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    1/M & 0 \\
    0 & 0 \\
    0 & -\frac{\ell Mg}{J}
    \end{bmatrix}
    $$

    En remplaçant $J=\tfrac{M\ell^2}{3}$, on obtient :

    $$
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    1/M & 0 \\
    0 & 0 \\
    0 & -3g/\ell
    \end{bmatrix}
    $$

    Les équations linéarisées s’écrivent alors :

    $$
    \begin{aligned}
    \Delta\dot{x} &= \Delta\dot{x},\\
    \Delta\ddot{x} &= -g\,\Delta\theta - g\,\Delta\phi,\\
    \Delta\dot{y} &= \Delta\dot{y},\\
    \Delta\ddot{y} &= \tfrac{1}{M}\,\Delta f,\\
    \Delta\dot{\theta} &= \Delta\dot{\theta},\\
    \Delta\ddot{\theta} &= -\tfrac{3g}{\ell}\,\Delta\phi.
    \end{aligned}
    $$
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


@app.cell
def _(M, g, l, np):

    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])

    B = np.array([
        [0, 0],
        [0, -g],
        [0, 0],
        [1/M, 0],
        [0, 0],
        [0, -3*g/l]
    ])

    print("Matrice A :")
    print(A)
    print("\nMatrice B :")
    print(B)
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell
def _(A, la, np):
    eigenvalues = la.eigvals(A)
    print("Valeurs propres de A :")
    print(eigenvalues)

    if np.all(np.real(eigenvalues) < 0):
        print("\nL'équilibre est asymptotiquement stable (toutes les valeurs propres ont des parties réelles strictement négatives).")
    else:
        print("\nL'équilibre n'est PAS asymptotiquement stable (certaines valeurs propres ont des parties réelles non négatives).")
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


@app.cell
def _(A, B, la, np):
    n = A.shape[0] 
    C = B
    A_power_B = B 
    for i in range(1, n):
        A_power_B = A @ A_power_B
        C = np.hstack((C, A_power_B))

    print("Matrice de commandabilité C (système complet) :")

    rank_C = la.matrix_rank(C)
    print(f"\nRang de C : {rank_C}")
    print(f"Dimension de l'état n : {n}")

    if rank_C == n:
        print("\nLe système COMPLET (6 états, 2 entrées) est commandable.")
    else:
        print("\nLe système COMPLET n'est PAS commandable.")
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
def _(g, l, la, np):
    A_lat = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B_lat = np.array([
        [0],
        [-g],
        [0],
        [-3*g/l]
    ])

    print("Matrice A_lat :")
    print(A_lat)
    print("\nMatrice B_lat :")
    print(B_lat)


    def calculate_controllability_lateral(A, B):
        """Calculates the controllability matrix and its rank for the lateral system."""
        n = A.shape[0]
        C = B
        A_power_B = B
        for i in range(1, n):
            A_power_B = A @ A_power_B
            C = np.hstack((C, A_power_B))
        return C, la.matrix_rank(C)


    C_lat, rank_C_lat = calculate_controllability_lateral(A_lat, B_lat)
    print("\nMatrice de commandabilité C_lat (système latéral) :")
    print(C_lat)
    n_lat = A_lat.shape[0] 
    print(f"\nRang de C_lat : {rank_C_lat}")
    print(f"Dimension de l'état latéral n_lat : {n_lat}")

    if rank_C_lat == n_lat:
        print("\nLe système LATÉRAL (4 états, entrée phi seulement) est commandable.")
    else:
        print("\nLe système LATÉRAL n'est PAS commandable.")
    return A_lat, B_lat


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


@app.cell
def _(l, mo, np, plt):
    t_span_linear = [0.0, 5.0]
    t_linear = np.linspace(t_span_linear[0], t_span_linear[1], 100)

    delta_y0 = 10.0 - l
    delta_theta0 = np.pi / 4 

    y_linear_t = (l + delta_y0) * np.ones_like(t_linear)
    theta_linear_t = delta_theta0 * np.ones_like(t_linear) 

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t_linear, y_linear_t, label=r"$y(t)$ (modèle linéaire)")
    axes[0].plot(t_linear, l * np.ones_like(t_linear), color="grey", ls="--", label=r"$y=\ell$")
    axes[0].set_ylabel("Hauteur (m)")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title("Modèle Linéarisé sans Contrôle ($\Delta f=0, \Delta \phi=0$)")
    axes[0].set_ylim(0, 11) 

    axes[1].plot(t_linear, theta_linear_t, label=r"$\theta(t)$ (modèle linéaire)")
    axes[1].plot(t_linear, np.zeros_like(t_linear), color="grey", ls="--", label=r"$\theta=0$")
    axes[1].set_ylabel(r"$\theta$ (rad)")
    axes[1].set_xlabel("temps $t$")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_ylim(-0.1, np.pi/4 + 0.1) 

    plt.tight_layout()

    print("Les graphiques montrent que selon le modèle linéarisé sans entrée de commande, la hauteur y(t) et l'angle d'inclinaison theta(t) restent constants à leurs écarts initiaux non nuls (10.0 m et pi/4 rad respectivement). Ceci est dû au fait que la dynamique linéarisée pour y et theta, lorsque les entrées sont nulles (\\Delta f = \\Delta \\phi = 0), est simplement \\Delta \\ddot{y} = 0 et \\Delta \\ddot{\\theta} = 0. Avec des vitesses initiales nulles (\\Delta \\dot{y}(0) = 0, \\Delta \\dot{\\theta}(0) = 0), cela signifie que les écarts de position et d'angle initiaux persistent. Cela confirme que l'équilibre n'est pas asymptotiquement stable – les perturbations ne se désintègrent pas sans contrôle.")

    mo.center(fig)
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


@app.cell
def _(A_lat, B_lat, mo, np, plt, scipy):
    def solve_linear_manual(t_span, y0, A_cl):
        def linear_dynamics(t, y):
            return A_cl @ y
        sol = scipy.integrate.solve_ivp(linear_dynamics, t_span, y0, dense_output=True)
        return sol.sol


    K_manual = np.array([0, 0, -1/75, -2/15]) 


    delta_zlat0 = np.array([0.0, 0.0, np.pi/4, 0.0])
    t_span_manual = [0.0, 30.0]
    t_manual = np.linspace(t_span_manual[0], t_span_manual[1], 300)


    A_cl_manual = A_lat - B_lat @ K_manual.reshape(1, -1)

    sol_manual = solve_linear_manual(t_span_manual, delta_zlat0, A_cl_manual)
    delta_zlat_t = sol_manual(t_manual)

    delta_theta_t = delta_zlat_t[2]
    delta_dtheta_t = delta_zlat_t[3]
    delta_phi_t = - K_manual @ delta_zlat_t

    fig_manual_final, axes_manual_final = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes_manual_final[0].plot(t_manual, delta_theta_t, label=r"$\Delta\theta(t)$ (rad)")
    axes_manual_final[0].plot(t_manual, np.zeros_like(t_manual), color="grey", ls="--")
    axes_manual_final[0].plot(t_manual, np.full_like(t_manual, np.pi/2), color="red", ls=":", label=r"$+\pi/2$")
    axes_manual_final[0].plot(t_manual, np.full_like(t_manual, -np.pi/2), color="red", ls=":", label=r"$-\pi/2$")
    axes_manual_final[0].set_ylabel(r"$\Delta\theta$ (rad)")
    axes_manual_final[0].set_title("Dynamique Latérale Linéarisée avec Contrôle Manuel (Pôles à -0.2)")
    axes_manual_final[0].grid(True)
    axes_manual_final[0].legend()
    axes_manual_final[1].plot(t_manual, delta_dtheta_t, label=r"$\Delta\dot{\theta}(t)$ (rad/s)")
    axes_manual_final[1].plot(t_manual, np.zeros_like(t_manual), color="grey", ls="--")
    axes_manual_final[1].set_ylabel(r"$\Delta\dot{\theta}$ (rad/s)")
    axes_manual_final[1].grid(True)
    axes_manual_final[1].legend()
    axes_manual_final[2].plot(t_manual, delta_phi_t, label=r"$\Delta\phi(t)$ (rad)")
    axes_manual_final[2].plot(t_manual, np.zeros_like(t_manual), color="grey", ls="--")
    axes_manual_final[2].plot(t_manual, np.full_like(t_manual, np.pi/2), color="red", ls=":", label=r"$+\pi/2$")
    axes_manual_final[2].plot(t_manual, np.full_like(t_manual, -np.pi/2), color="red", ls=":", label=r"$-\pi/2$")
    axes_manual_final[2].set_ylabel(r"$\Delta\phi$ (rad)")
    axes_manual_final[2].set_xlabel("time $t$")
    axes_manual_final[2].grid(True)
    axes_manual_final[2].legend()
    plt.tight_layout()
    print(f"\nDelta theta initial : {delta_theta_t[0]:.4f} rad")
    print(f"Max absolu Delta theta : {np.max(np.abs(delta_theta_t)):.4f} rad")
    print(f"Delta phi initial : {delta_phi_t[0]:.4f} rad")
    print(f"Max absolu Delta phi : {np.max(np.abs(delta_phi_t)):.4f} rad")


    theta_at_20s = sol_manual(20.0)[2]
    print(f"Delta theta à 20s : {theta_at_20s:.4f} rad (cible 0)")


    print("\nmodèle en boucle fermée est-il asymptotiquement stable ? Non.")
    print("Les valeurs propres de la matrice complète en boucle fermée (A_lat - B_lat*K) sont s^2 (s+0.2)^2 = 0, ce qui signifie qu'il y a des valeurs propres à 0.")
    print("Plus précisément, les états x et dx ne sont pas directement contrôlés par ce K et ne sont affectés indirectement que par theta/dtheta.")
    print("Comme delta_theta et delta_dtheta décroissent, delta_ddot_x approche 0, ce qui signifie que delta_dot_x tend vers une constante et delta_x dérive linéairement.")
    print("C'est acceptable selon l'énoncé du problème ('ne pas se soucier d'une possible dérive de delta_x'), mais cela signifie que le système n'est pas asymptotiquement stable au sens strict (les états ne retournent pas tous à leurs valeurs d'équilibre). Il est stable par rapport à theta, dtheta et dx (ils retournent à 0).")

    mo.center(fig_manual_final)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Essayons de trouver les deux coefficients manquants de la matrice

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{1\times 4}
    $$

    telle que la loi de commande

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

    gère lorsque
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  et $\dot{\theta}(0) =0$ pour :

      - faire tendre $\Delta \theta(t) \to 0$ en approximativement 20 secondes (ou moins),
      - $|\Delta \theta(t)| < \pi/2$ et $|\Delta \phi(t)| < \pi/2$ à tout moment,
      - (mais nous ne nous soucions pas d'une possible dérive de $\Delta x(t)$).

    Cela dit,


    Nous contrôlons le système latéral avec l'état $z_{lat} = [\Delta x, \Delta \dot{x}, \Delta \theta, \Delta \dot{\theta}]^T$ en utilisant l'entrée $\Delta \phi = -K z_{lat}$.
    La matrice de gain $K$ est de la forme spécifiée : $K = [0, 0, K_3, K_4]$.
    La loi de commande est donc $\Delta \phi = - (0 \cdot \Delta x + 0 \cdot \Delta \dot{x} + K_3 \Delta \theta + K_4 \Delta \dot{\theta}) = -K_3 \Delta \theta - K_4 \Delta \dot{\theta}$.

    Le système latéral en boucle fermée est décrit par l'équation $\Delta \dot{z}_{lat} = (A_{lat} - B_{lat}K) \Delta z_{lat}$. La matrice du système en boucle fermée est :

    $$
    A_{cl\_manual} = A_{lat} - B_{lat}K = \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix} - 
    \begin{bmatrix}
    0 \\ -g \\ 0 \\ -3g/\ell
    \end{bmatrix} \begin{bmatrix} 0 & 0 & K_3 & K_4 \end{bmatrix}
    = \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix} - 
    \begin{bmatrix}
    0 & 0 & 0 & 0 \\
    0 & 0 & -gK_3 & -gK_4 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & -3g/\ell K_3 & -3g/\ell K_4
    \end{bmatrix}
    $$

    $$
    A_{cl\_manual} = \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g + gK_3 & gK_4 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 3g/\ell K_3 & 3g/\ell K_4
    \end{bmatrix}
    $$

    Les équations d'état correspondantes sont :
    $\Delta \dot{x} = \Delta \dot{x}$
    ,
    $\Delta \ddot{x} = g(K_3-1)\Delta \theta + g K_4 \Delta \dot{\theta}$
    ,
    $\Delta \dot{\theta} = \Delta \dot{\theta}$
    ,
    $\Delta \ddot{\theta} = (3g/\ell) K_3 \Delta \theta + (3g/\ell) K_4 \Delta \dot{\theta}$

    Notez que les équations pour $\Delta \theta$ et $\Delta \dot{\theta}$ forment un système de second ordre indépendant si l'on ne se soucie pas de $\Delta x$ et $\Delta \dot{x}$:
    $\begin{bmatrix} \Delta \dot{\theta} \\ \Delta \ddot{\theta} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ (3g/\ell) K_3 & (3g/\ell) K_4 \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta \dot{\theta} \end{bmatrix}$.
    L'équation caractéristique de ce sous-système est $\det \begin{bmatrix} -\lambda & 1 \\ (3g/\ell) K_3 & (3g/\ell) K_4 -\lambda \end{bmatrix} = -\lambda((3g/\ell) K_4 - \lambda) - (3g/\ell) K_3 = \lambda^2 - (3g/\ell) K_4 \lambda - (3g/\ell) K_3 = 0$.

    Nous voulons que $\Delta \theta(t) \to 0$ en approximativement 20 secondes. Le temps de stabilisation ($T_s$) d'un système est grossièrement lié à la partie réelle la plus grande de ses pôles ($|\text{Re}(p_{max})|$), typiquement $T_s \approx 4/|\text{Re}(p_{max})|$ pour un critère à 2%. Pour $T_s=20$s, nous visons une partie réelle de pôle autour de $-4/20 = -0.2$. Pour obtenir une convergence sans dépassement (ce qui est souvent souhaitable pour un atterrissage), on choisit des pôles réels égaux et négatifs (amortissement critique).

    **Démarche de réflexion pour le réglage manuel :**

    1.  On Identifie le sous-système à contrôler en choisissant $K$ sous la forme $[0, 0, K_3, K_4]$, nous décidons de n'utiliser le retour d'état que sur $\Delta \theta$ et $\Delta \dot{\theta}$. Les équations du système en boucle fermée montrent que la dynamique de $\Delta \theta$ et $\Delta \dot{\theta}$ forme un sous-système de second ordre qui est contrôlé indépendamment (c'est un "pendule inversé" stabilisé).
    2.  On détermine l'équation caractéristique du sous-système $\theta$ comme calculé ci-dessus, c'est $\lambda^2 - (3g/\ell) K_4 \lambda - (3g/\ell) K_3 = 0$.
    3. On choisit les pôles voulus pour la dynamique de $\Delta \theta$ puisque l'objectif est un temps de stabilisation de $\sim 20$s. Cela nous amène à choisir des pôles avec une partie réelle autour de $-0.2$. Pour une réponse sans dépassement, nous choisissons deux pôles réels et égaux : $p_1 = p_2 = -0.2$.
    4.  Nous déterminons le polynôme caractéristique correspondant aux pôles $-0.2, -0.2$ est $(s - (-0.2))(s - (-0.2)) = (s + 0.2)^2 = s^2 + 0.4s + 0.04 = 0$.
    5.  Nous faisons correspondre les coefficients pour trouver $K_3$ et $K_4$ en comparant les coefficients de l'équation caractéristique du sous-système $\theta$ et du polynôme désiré :
        $\lambda^2 \underbrace{- (3g/\ell) K_4}_{\text{coeff de }\lambda} \lambda \underbrace{- (3g/\ell) K_3}_{\text{coeff constant}} = 0$
        $s^2 \underbrace{+ 0.4}_{\text{coeff de }s} s \underbrace{+ 0.04}_{\text{coeff constant}} = 0$
        (En utilisant $s$ pour les pôles dans le polynôme désiré correspond à $\lambda$ dans l'équation caractéristique du système).
        En identifiant les coefficients, on obtient :
        $-(3g/\ell) K_4 = 0.4 \implies K_4 = -0.4 / (3g/\ell)$.
        $-(3g/\ell) K_3 = 0.04 \implies K_3 = -0.04 / (3g/\ell)$.
        En utilisant les constantes $g=1$ et $\ell=1$, on a $3g/\ell = 3$.
        $K_4 = -0.4 / 3 = -2/15$.
        $K_3 = -0.04 / 3 = -1/75$.
        Donc la matrice de gain manuelle est $K = [0, 0, -1/75, -2/15]$.

    6.  Vérifions les contraintes :
        *   Convergence de $\Delta \theta$: Par construction (placement de pôles à -0.2), $\Delta \theta(t)$ devrait tendre vers 0 en environ 20s.
        *   $|\Delta \theta(t)| < \pi/2$: L'état initial est $\Delta \theta(0) = \pi/4$, ce qui est inférieur à $\pi/2$. Avec des pôles réels négatifs, il n'y aura pas de dépassement, donc $|\Delta \theta(t)| \le |\Delta \theta(0)| = \pi/4 < \pi/2$ pour une simulation linéaire.
        *   $|\Delta \phi(t)| < \pi/2$: L'entrée de commande est $\Delta \phi(t) = -K_3 \Delta \theta(t) - K_4 \Delta \dot{\theta}(t)$. L'entrée initiale est $\Delta \phi(0) = -K_3 \Delta \theta(0) - K_4 \Delta \dot{\theta}(0) = -(-1/75)(\pi/4) - (-2/15)(0) = \pi/300 \approx 0.0105$ rad. C'est très inférieur à $\pi/2 \approx 1.57$ rad. Les simulations (voir code ci-dessous) montrent que $|\Delta \phi(t)|$ reste dans les limites.

    7.  Analysons la stabilité du système complet en boucle fermée : Pour déterminer la stabilité asymptotique du système complet en boucle fermée

    $$
    \Delta \dot{z}_{lat} = A_{cl\_manual} \Delta z_{lat}
    $$

    il faut vérifier les valeurs propres de $A_{cl\_manual}$. Le polynôme caractéristique du système complet est $\det(sI - A_{cl\_manual})$.

    $$
    \det(sI - A_{cl\_manual}) = \det \begin{bmatrix}
    s & -1 & 0 & 0 \\
    0 & s & g - gK_3 & -gK_4 \\
    0 & 0 & s & -1 \\
    0 & 0 & -3g/\ell K_3 & s - 3g/\ell K_4
    \end{bmatrix}
    $$

        En développant le déterminant (par blocs ou par cofacteurs), on voit qu'il y a un facteur $s^2$ provenant du bloc supérieur gauche (lié aux dynamiques de $x, \dot{x}$). Le polynôme caractéristique est $s^2 (s^2 - (3g/\ell) K_4 s - (3g/\ell) K_3)$.
        Avec notre choix de $K_3$ et $K_4$, cela devient $s^2 (s^2 + 0.4s + 0.04) = s^2 (s+0.2)^2 = 0$.
        Les valeurs propres sont $0, 0, -0.2, -0.2$. Comme il y a des valeurs propres avec une partie réelle nulle (les deux 0), le système complet en boucle fermée n'est **pas** asymptotiquement stable au sens strict (tous les états ne retournent pas à zéro). Les états $\Delta \theta$ et $\Delta \dot{\theta}$ retournent à zéro, mais la dynamique de $\Delta x$ et $\Delta \dot{x}$ ne tend pas nécessairement vers zéro. $\Delta \ddot{x}$ tend vers zéro, donc $\Delta \dot{x}$ tend vers une constante, et $\Delta x$ dérive linéairement. Cependant, l'énoncé spécifie que "we don't care about a possible drift of $\Delta x(t)$", donc ce comportement est acceptable pour cette tâche spécifique.  Le système est stable pour les états $\theta, \dot{\theta}, \dot{x}$.

    *   **Essai 1 :** Choisir des pôles plus rapides, par exemple $p_1 = p_2 = -0.5$. Cela donnerait un temps de stabilisation de $\sim 4/0.5 = 8$s pour $\Delta \theta$. $K_4 = (-1)/3 = -1/3$, $K_3 = -0.25/3 = -1/12$. La simulation d'avant montre que cela fonctionne aussi et respecte les contraintes de $\phi$, mais c'est plus rapide que les 20s demandés.
    *   **Essai 2 :** Choisir des pôles à $-0.2, -0.2$. $K_4 = -0.4/3 = -2/15$, $K_3 = -0.04/3 = -1/75$. La simulation d'avant montre que $\Delta \theta$ se stabilise en environ 20s et que $\Delta \phi$ reste bien dans les limites. Ce $K$ correspond le mieux aux exigences.
    """
    )
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


if __name__ == "__main__":
    app.run()
