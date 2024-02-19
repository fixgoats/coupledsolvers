use nalgebra::{SVector, Vector2, SMatrix};
use rand::Rng;
use plotters::prelude::*;
use ode_solvers::rk4::*;
use num::Complex;

const NSITES: usize = 1;
const TRIPLENSITES: usize = 3 * NSITES;
// type C64 = Complex<f64>;
type State = SVector<f64, TRIPLENSITES>;
/*type SimplerState = Vector2<f64>;

struct FallingBall;

impl ode_solvers::System<f64, SimplerState> for FallingBall {
    fn system(&self, _t: f64, y: &SimplerState, dy: &mut SimplerState) {
        dy[0] = y[1];
        dy[1] = -9.8;
    }
}*/

struct Condensate {
    // j: SMatrix<f64, NSITES, NSITES>,
    // d: SMatrix<f64, NSITES, NSITES>,
    alpha: f64,
    omega: f64,
    g: f64,
    r: f64,
    // beta: f64,
    // v: f64,
    gamma: f64,
    p: f64,
}

fn norm_sqr(x: f64, y: f64) -> f64 {
    return x * x + y * y; 
}

impl ode_solvers::System<f64, State> for Condensate {
    fn system(&self, _x: f64, y: &State, dy: &mut State) {
        for i in 0..NSITES {
            let x = y[2*NSITES+i];
            let psir = y[2*i];
            let psii = y[2*i+1];
            dy[2*i] = 0.5 * self.r * x * psir 
                -(self.omega + self.g * x 
                   + self.alpha * norm_sqr(psir, psii)) * psii;
            dy[2*i+1] = 0.5 * self.r * x * psii
                + (self.omega + self.g * x
                   + self.alpha * norm_sqr(psii, psir)) * psir;
            dy[2*NSITES+i] = -(self.gamma + self.r * norm_sqr(psir, psii)) * x + self.p;
        }
    }
}

#[allow(dead_code)]
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let mut y0 = SVector::<f64, TRIPLENSITES>::zeros();
    for i in 0..TRIPLENSITES {
        y0[i] = rng.gen_range(0.0..2.);
    }

    let system = Condensate {
        alpha: 0.01,
        omega: 0.01,
        g: 0.01,
        r: 0.1,
        gamma: 0.01,
        p: 30.
    };

    /*let system = FallingBall;
    let y0 = SimplerState::new(10.0, 0.);*/
    let startt = 0.;
    let endt = 2.;
    let dt = 1e-4;
    let nsteps = ((endt - startt) / dt) as usize;
    let mut bleh = Rk4::new(system, startt,  y0, endt, dt);
    let res = bleh.integrate();
    let psisq = bleh.y_out().

    let area = BitMapBackend::new("ha.png", (1024, 760)).into_drawing_area();
    area.fill(&WHITE)?;
    let minx = startt;
    let maxx = endt;
    let miny = 0.0;
    let maxy = 0.01;
    let mut chart = ChartBuilder::on(&area)
        .margin(20)
        .caption("psi?".to_string(), ("sans", 20))
        .build_cartesian_2d((minx..maxx).step(0.001), (miny..maxy).step(0.001))?;

    chart.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .max_light_lines(4)
        .draw()?;

    chart.draw_series(
        LineSeries::new(
            (0..nsteps).map(|i| (i as f64 * dt, norm_sqr(bleh.y_out()[i][0], bleh.y_out()[i][1]))),
            &BLACK
        ))?
        .label("Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));
    chart.draw_series(
        LineSeries::new(
            (0..nsteps).map(|i| (i as f64 * dt, bleh.y_out()[i][2])),
            &BLUE
        ))?
        .label("Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart.configure_series_labels().border_style(BLACK).draw()?;
    area.present().expect("Can't write to file");
    Ok(())
}
