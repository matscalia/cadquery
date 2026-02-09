from typing import Tuple, Union, Any, Callable, List, Optional, Iterable, Dict, Sequence
from typing import Literal
from numpy.typing import NDArray as Array
from numpy import float64 as Float, linspace, trapezoid, vectorize
from itertools import accumulate, chain
from math import atan2, pi, sin, cos, radians, sqrt

from numpy import array, full, inf, sign
from numpy.linalg import norm
import nlopt

from OCP.gp import gp_Vec2d

from .shapes import Geoms
from ..types import Real

NoneType = type(None)

SegmentDOF = Tuple[float, float, float, float]  # p1 p2
ArcDOF = Tuple[float, float, float, float, float]  # p r a da
EllipseDOF = Tuple[
    float, float, float, float, float, float, float
]  # cx, cy, xr, yr, rot, t, dt
DOF = Union[SegmentDOF, ArcDOF, EllipseDOF]

ConstraintKind = Literal[
    "Fixed",
    "FixedPoint",
    "Coincident",
    "Angle",
    "Length",
    "Distance",
    "Radius",
    "Orientation",
    "ArcAngle",
    "PointOnObject",
    "Equal",
    "EqualRadius",
]

ConstraintInvariants = {  # (arity, geometry types, param type, conversion func)
    "Fixed": (1, ("CIRCLE", "LINE", "ELLIPSE"), NoneType, None),
    "FixedPoint": (1, ("CIRCLE", "LINE", "ELLIPSE"), Optional[Real], None),
    "Coincident": (2, ("CIRCLE", "LINE", "ELLIPSE"), NoneType, None),
    "Angle": (2, ("CIRCLE", "LINE", "ELLIPSE"), Real, radians),
    "Length": (1, ("CIRCLE", "LINE", "ELLIPSE"), Real, None),
    "Distance": (
        2,
        ("CIRCLE", "LINE", "ELLIPSE"),
        Tuple[Optional[Real], Optional[Real], Real],
        None,
    ),
    "Radius": (1, ("CIRCLE", "ELLIPSE"), Real | Tuple[Literal[0, 1], Real], None),
    "Orientation": (1, ("LINE", "CIRCLE", "ELLIPSE"), Tuple[Real, Real], None),
    "ArcAngle": (1, ("CIRCLE", "ELLIPSE"), Real, radians),
    "PointOnObject": (2, ("CIRCLE", "LINE", "ELLIPSE"), Optional[Real], None),
    "Equal": (2, ("LINE", "CIRCLE", "ELLIPSE"), NoneType, None),
    "EqualRadius": (
        2,
        ("CIRCLE", "ELLIPSE"),
        NoneType | Tuple[Literal[0, 1] | None, Literal[0, 1] | None],
        None,
    ),
}

Constraint = Tuple[Tuple[int, Optional[int]], ConstraintKind, Optional[Any]]

DIFF_EPS = 1e-10
TOL = 1e-9
MAXITER = 0


def invalid_args(*t):

    return ValueError("Invalid argument types {t}")


def arc_first(x):

    return array((x[0] + x[2] * sin(x[3]), x[1] + x[2] * cos(x[3])))


def arc_last(x):

    return array((x[0] + x[2] * sin(x[3] + x[4]), x[1] + x[2] * cos(x[3] + x[4])))


def arc_point(x, val):

    if val is None:
        rv = x[:2]
    else:
        a = x[3] + val * x[4]
        rv = array((x[0] + x[2] * sin(a), x[1] + x[2] * cos(a)))

    return rv


def line_point(x, val):

    return x[:2] + val * (x[2:] - x[:2])


def arc_first_tangent(x):

    return gp_Vec2d(sign(x[4]) * cos(x[3]), -sign(x[4]) * sin(x[3]))


def arc_last_tangent(x):

    return gp_Vec2d(sign(x[4]) * cos(x[3] + x[4]), -sign(x[4]) * sin(x[3] + x[4]))


def ellipse_point(x, val):
    if val == None:
        return x[:2]

    cx, cy, xr, yr, rot, t, dt = x
    param = t + val * dt
    local_x = xr * cos(param)
    local_y = yr * sin(param)
    rx = local_x * cos(rot) - local_y * sin(rot)
    ry = local_x * sin(rot) + local_y * cos(rot)
    return array((cx + rx, cy + ry))


def ellipse_point_by_angle(x, angle):
    cx, cy, xr, yr, rot, _, _ = x

    radius = (
        lambda theta: xr * yr / sqrt((yr * cos(theta)) ** 2 + (xr * sin(theta)) ** 2)
    )
    local_x = radius(angle) * cos(angle)
    local_y = radius(angle) * sin(angle)
    rx = local_x * cos(rot) - local_y * sin(rot)
    ry = local_x * sin(rot) + local_y * cos(rot)
    return array((cx + rx, cy + ry))


def ellipse_first(x):
    return ellipse_point(x, 0)


def ellipse_last(x):
    return ellipse_point(x, 1)


def ellipse_first_tangent(x):
    cx, cy, xr, yr, rot, t, dt = x
    direction = sign(dt)
    local_dx = -direction * xr * sin(t)
    local_dy = direction * yr * cos(t)
    rdx = local_dx * cos(rot) - local_dy * sin(rot)
    rdy = local_dx * sin(rot) - local_dy * cos(rot)
    tangent = array([rdx, rdy])
    tangent = tangent / norm(tangent)
    return gp_Vec2d(tangent[0], tangent[1])


def ellipse_last_tangent(x):
    cx, cy, xr, yr, rot, t, dt = x
    direction = sign(dt)
    t1 = t + dt
    local_dx = -direction * xr * sin(t1)
    local_dy = direction * yr * cos(t1)
    rdx = local_dx * cos(rot) - local_dy * sin(rot)
    rdy = local_dx * sin(rot) - local_dy * cos(rot)
    tangent = array([rdx, rdy])
    tangent = tangent / norm(tangent)
    return gp_Vec2d(tangent[0], tangent[1])


ELLIPSE_LEN_SAMPLES = 1000


def ellipse_length(x):
    """
    Temporary, very inefficient implementation of the elliptic integral
    """
    cx, cy, xr, yr, rot, t, dt = x
    samples = ELLIPSE_LEN_SAMPLES
    theta = linspace(t, t + dt, samples)
    integrand = lambda t: sqrt(xr ** 2 * cos(t) ** 2 + yr ** 2 * sin(t) ** 2)
    integrand = vectorize(integrand)
    integral = trapezoid(integrand(theta), theta)
    while True:
        tmp = integral
        samples *= 2
        theta = linspace(t, t + dt, samples)
        integral = trapezoid(integrand(theta), theta)
        if abs(integral - tmp) < DIFF_EPS:
            break
    return integral


def ellipse_angle(x, val):
    p = ellipse_point(x, val)
    return atan2(p[1], p[0])


def fixed_cost(x, t, x0, val):

    return norm(x - x0)


def fixed_point_cost(x, t, x0, val):

    if t == "LINE":
        rv = norm(line_point(x, val) - line_point(x0, val))
    elif t == "CIRCLE":
        rv = norm(arc_point(x, val) - arc_point(x0, val))
    elif t == "ELLIPSE":
        rv = norm(ellipse_point(x, val) - ellipse_point(x0, val))
    else:
        raise invalid_args(t)

    return rv


def coincident_cost(x1, t1, x10, x2, t2, x20, val):
    v1 = (0, 0)
    v2 = (0, 0)
    if t1 == "LINE":
        v1 = x1[2:]
    elif t1 == "CIRCLE":
        v1 = arc_last(x1)
    elif t1 == "ELLIPSE":
        v1 = ellipse_point(x1, 1)
    else:
        raise invalid_args(t1, t2)

    if t2 == "LINE":
        v2 = x2[:2]
    elif t2 == "CIRCLE":
        v2 = arc_first(x2)
    elif t2 == "ELLIPSE":
        v2 = ellipse_point(x2, 0)
    else:
        raise invalid_args(t1, t2)

    return norm(v1 - v2)


def angle_cost(x1, t1, x10, x2, t2, x20, val):
    v1 = (0, 0)
    v2 = (0, 0)
    if t1 == "LINE":
        v1 = gp_Vec2d(*(x1[2:] - x1[:2]))
    elif t1 == "CIRCLE":
        v1 = arc_last_tangent(x1)
    elif t1 == "ELLIPSE":
        v1 = ellipse_last_tangent(x1)
    else:
        raise invalid_args(t1, t2)

    if t2 == "LINE":
        v2 = gp_Vec2d(*(x2[2:] - x2[:2]))
    elif t2 == "CIRCLE":
        v2 = arc_first_tangent(x2)
    elif t2 == "ELLIPSE":
        v2 = ellipse_first_tangent(x2)
    else:
        raise invalid_args(t1, t2)

    return v2.Angle(v1) - val


def length_cost(x, t, x0, val):

    rv = 0

    if t == "LINE":
        rv = norm(x[2:] - x[:2]) - val
    elif t == "CIRCLE":
        rv = norm(x[2] * x[4]) - val
    elif t == "ELLIPSE":
        rv = ellipse_length(x) - val
    else:
        raise invalid_args(t)

    return rv


def distance_cost(x1, t1, x10, x2, t2, x20, val):

    val1, val2, d = val

    if t1 == "LINE":
        v1 = line_point(x1, val1)
    elif t1 == "CIRCLE":
        v1 = arc_point(x1, val1)
    elif t1 == "ELLIPSE":
        v1 = ellipse_point(x1, val1)
    else:
        invalid_args(t1, t2)

    if t2 == "LINE":
        v2 = line_point(x2, val2)
    elif t2 == "CIRCLE":
        v2 = arc_point(x2, val2)
    elif t2 == "ELLIPSE":
        v2 = ellipse_point(x2, val2)
    else:
        invalid_args(t1, t2)

    return norm(v1 - v2) - d


def ellipse_radius_val(v):
    return (
        isinstance(v, tuple)
        and len(v) == 2
        and isinstance(v[0], int)
        and isinstance(v[1], Real)
        and (v[0] == 1 or v[0] == 0)
    )


def radius_cost(x, t, x0, val):

    if t == "CIRCLE" and isinstance(val, Real):
        rv = x[2] - val
    elif t == "ELLIPSE" and ellipse_radius_val(val):
        cx, cy, xr, yr, rot, t, dt = x
        r = (xr, yr)
        rv = r[val[0]] - val[1]
    else:
        raise invalid_args(t, val)

    return rv


def orientation_cost(x, t, x0, val):

    if t == "LINE":
        rv = gp_Vec2d(*(x[2:] - x[:2])).Angle(gp_Vec2d(*val))
    elif t == "ELLIPSE":
        cx, cy, xr, yr, rot, t, dt = x
        rv = gp_Vec2d(cos(rot), sin(rot)).Angle(gp_Vec2d(*val))
    else:
        raise invalid_args(t)

    return rv


def arc_angle_cost(x, t, x0, val):

    if t == "CIRCLE":
        rv = x[4] - val
    elif t == "ELLIPSE":
        a = ellipse_angle(x, 0)
        a1 = ellipse_angle(x, 1)
        rv = (a1 - a) - val
    else:
        raise invalid_args(t)

    return rv


def equal_cost(x1, t1, x10, x2, t2, x20, val):
    if t1 == "LINE":
        length1 = norm(x1[2:] - x1[:2])
    elif t1 == "CIRCLE":
        length1 = norm(x1[2] * x1[4])
    elif t1 == "ELLIPSE":
        length1 = ellipse_length(x1)

    if t2 == "LINE":
        length2 = norm(x2[2:] - x2[:2])
    elif t2 == "CIRCLE":
        length2 = norm(x2[2] * x2[4])
    elif t2 == "ELLIPSE":
        length2 = ellipse_length(x2)

    return length1 - length2


def equal_radius_cost(x1, t1, x10, x2, t2, x20, val):
    if t1 == "CIRCLE" and (val == None or val[0] == None):
        r1 = x1[2]
    elif (
        t1 == "ELLIPSE"
        and isinstance(val, tuple)
        and len(val) == 2
        and (val[0] == 1 or val[0] == 0)
    ):
        cx, cy, xr, yr, rot, t, dt = x1
        r = (xr, yr)
        r1 = r[val[0]]
    else:
        raise invalid_args(t1, t2, val)

    if t2 == "CIRCLE" and (val == None or val[1] == None):
        r2 = x2[2]
    elif (
        t2 == "ELLIPSE"
        and isinstance(val, tuple)
        and len(val) == 2
        and (val[1] == 1 or val[1] == 0)
    ):
        cx, cy, xr, yr, rot, t, dt = x2
        r = (xr, yr)
        r2 = r[val[1]]
    else:
        raise invalid_args(t1, t2, val)

    return r1 - r2


def point_on_object_cost(x1, t1, x10, x2, t2, x20, val):

    if t1 == "LINE" and val == None:
        raise invalid_args(t1, val)

    if t1 == "LINE":
        p = line_point(x1, val)
    elif t1 == "CIRCLE" and val == None:
        p = x1[:2]
    elif t1 == "CIRCLE":
        p = arc_point(x1, val)
    elif t1 == "ELLIPSE":
        p = ellipse_point(x1, val)

    if t2 == "LINE":
        start = x2[:2]
        end = x2[2:]
        v = end - start
        l = norm(v)
        d = p - start
        return (v[0] * d[1] - v[1] * d[0]) / l
    elif t2 == "CIRCLE":
        c = x2[:2]
        radius = x2[2]
        return norm(p - c) - radius
    elif t2 == "ELLIPSE":
        cx, cy, xr, yr, rot, t, _ = x2
        fx2 = (cx, cy, xr, yr, rot, 0, 2 * pi)
        c = array(cx, cy)
        pdir = p - c
        pdir = pdir / norm(pdir)
        angle = atan2(pdir[1], pdir[0])
        ep = ellipse_point_by_angle(fx2, angle)
        return norm(p - ep)


# dictionary of individual constraint cost functions
costs: Dict[str, Callable[..., float]] = dict(
    Fixed=fixed_cost,
    FixedPoint=fixed_point_cost,
    Coincident=coincident_cost,
    Angle=angle_cost,
    Length=length_cost,
    Distance=distance_cost,
    Radius=radius_cost,
    Orientation=orientation_cost,
    ArcAngle=arc_angle_cost,
    PointOnObject=point_on_object_cost,
    Equal=equal_cost,
    EqualRadius=equal_radius_cost,
)


class SketchConstraintSolver(object):

    entities: List[DOF]
    constraints: List[Constraint]
    geoms: List[Geoms]
    ne: int
    nc: int
    ixs: List[int]

    def __init__(
        self,
        entities: Iterable[DOF],
        constraints: Iterable[Constraint],
        geoms: Iterable[Geoms],
    ):

        self.entities = list(entities)
        self.constraints = list(constraints)
        self.geoms = list(geoms)

        self.ne = len(self.entities)
        self.nc = len(self.constraints)

        # validate and transform constraints

        # indices of x corresponding to the entities
        self.ixs = [0] + list(accumulate(len(e) for e in self.entities))

    def _cost(
        self, x0: Array[Float]
    ) -> Tuple[
        Callable[[Array[Float]], float],
        Callable[[Array[Float], Array[Float]], None],
        Array[Float],
        Array[Float],
    ]:

        ixs = self.ixs
        constraints = self.constraints
        geoms = self.geoms

        # split initial values per entity
        x0s = [x0[ixs[e] : ixs[e + 1]] for e in range(self.ne)]

        def f(x) -> float:
            """
            Cost function to be minimized
            """

            rv = 0.0

            for i, ((e1, e2), kind, val) in enumerate(constraints):

                cost = costs[kind]

                # build arguments for the specific constraint
                args = [x[ixs[e1] : ixs[e1 + 1]], geoms[e1], x0s[e1]]
                if e2 is not None:
                    args += [x[ixs[e2] : ixs[e2 + 1]], geoms[e2], x0s[e2]]

                # evaluate
                rv += cost(*args, val) ** 2

            return rv

        def grad(x, rv) -> None:
            """
            Gradient of the cost function
            """

            rv[:] = 0

            for i, ((e1, e2), kind, val) in enumerate(constraints):

                cost = costs[kind]

                # build arguments for the specific constraint
                x1 = x[ixs[e1] : ixs[e1 + 1]]
                args = [x1.copy(), geoms[e1], x0s[e1]]
                if e2 is not None:
                    x2 = x[ixs[e2] : ixs[e2 + 1]]
                    args += [x2.copy(), geoms[e2], x0s[e2]]

                # evaluate
                tmp = cost(*args, val)

                for j, k in enumerate(range(ixs[e1], ixs[e1 + 1])):
                    args[0][j] += DIFF_EPS
                    tmp1 = cost(*args, val)
                    rv[k] += 2 * tmp * (tmp1 - tmp) / DIFF_EPS
                    args[0][j] = x1[j]

                if e2 is not None:
                    for j, k in enumerate(range(ixs[e2], ixs[e2 + 1])):
                        args[3][j] += DIFF_EPS
                        tmp2 = cost(*args, val)
                        rv[k] += 2 * tmp * (tmp2 - tmp) / DIFF_EPS
                        args[3][j] = x2[j]

        # generate lower and upper bounds for optimization
        lb = full(ixs[-1], -inf)
        ub = full(ixs[-1], +inf)

        for i, g in enumerate(geoms):
            if g == "CIRCLE":
                lb[ixs[i] + 2] = 0  # lower bound for radius
            elif g == "ELLIPSE":
                lb[ixs[i] + 2] = 0
                lb[ixs[i] + 3] = 0

        return f, grad, lb, ub

    def solve(self) -> Tuple[Sequence[Sequence[float]], Dict[str, Any]]:

        x0 = array(list(chain.from_iterable(self.entities))).ravel()
        f, grad, lb, ub = self._cost(x0)

        def func(x, g):

            if g.size > 0:
                grad(x, g)

            return f(x)

        opt = nlopt.opt(nlopt.LD_SLSQP, len(x0))
        opt.set_min_objective(func)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

        opt.set_ftol_abs(0)
        opt.set_ftol_rel(0)
        opt.set_xtol_rel(TOL)
        opt.set_xtol_abs(TOL * 1e-3)
        opt.set_maxeval(MAXITER)

        x = opt.optimize(x0)
        status = {
            "entities": self.entities,
            "cost": opt.last_optimum_value(),
            "iters": opt.get_numevals(),
            "status": opt.last_optimize_result(),
        }

        ixs = self.ixs

        return [x[i1:i2] for i1, i2 in zip(ixs, ixs[1:])], status
