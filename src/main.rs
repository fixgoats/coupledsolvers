use nalgebra::{Vector2, Matrix2};
use ode_solvers::rk4::*;
use num::Complex;

type C64 = Complex<f64>;
type State = (Vector2<C64>, Vector2<f64>);

struct Condensate {
    j: Matrix2<f64>,
    d: Matrix2<f64>,
    alpha: f64,
    omega: f64,
    g: f64,
    r: f64,
    beta: f64,
    v: f64,
    gamma: f64,
    p: f64,
}

impl ode_solvers::System<f64, State> for Condensate {
    fn system(&self, _x: f64, y: &State, dy: &mut State) {
    }
}

fn main() {
    let y0 = 0.;
    let system = LorenzAttractor {
        sigma: 10.,
        beta: 8. / 3.,
        rho: 28.
    };

    let mut bleh = Rk4::new(system, 0.,  y0, 10., 1e-2);
    let res = bleh.integrate();

    println!("{}", bleh.y_out()[900]);
}
