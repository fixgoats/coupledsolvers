use nalgebra::Vector3;
use plotters::prelude::*;
use ode_solvers::rk4::*;

type State = Vector3<f64>;

struct LorenzAttractor {
    sigma: f64,
    beta: f64,
    rho: f64,
}

impl ode_solvers::System<f64, State> for LorenzAttractor {
    fn system(&self, _x: f64, y: &State, dy: &mut State) {
        dy[0] = self.sigma * (y[1] - y[0]);
        dy[1] = y[0] * (self.rho - y[2]) - y[1];
        dy[2] = y[0] * y[1] - self.beta * y[2];
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let y0 = State::new(2., 2., 2.);
    let system = LorenzAttractor {
        sigma: 10.,
        beta: 8. / 3.,
        rho: 28.
    };

    let mut bleh = Rk4::new(system, 0.,  y0, 100., 1e-3);
    let res = bleh.integrate();

    let minx = bleh.y_out().iter().map(|v| v[0]).reduce(f64::min).unwrap() - 0.1;
    let maxx = bleh.y_out().iter().map(|v| v[0]).reduce(f64::max).unwrap() + 0.1;
    let miny = bleh.y_out().iter().map(|v| v[1]).reduce(f64::min).unwrap() - 0.1;
    let maxy = bleh.y_out().iter().map(|v| v[1]).reduce(f64::max).unwrap() + 0.1;
    let minz = bleh.y_out().iter().map(|v| v[2]).reduce(f64::min).unwrap() - 0.1;
    let maxz = bleh.y_out().iter().map(|v| v[2]).reduce(f64::max).unwrap() + 0.1;


    let area = BitMapBackend::new("bleh.png", (1024, 760)).into_drawing_area();
    area.fill(&WHITE)?;
    let x_axis = (minx..maxx).step(0.1);
    let y_axis = (miny..maxy).step(0.1);
    let z_axis = (minz..maxz).step(0.1);
    let mut chart = ChartBuilder::on(&area)
        .caption("Lorenz attractor".to_string(), ("sans", 20))
        .build_cartesian_3d(x_axis.clone(), y_axis.clone(), z_axis.clone())?;

    chart.with_projection(|mut pb| {
        pb.yaw = 0.1;
        pb.scale = 0.9;
        pb.into_matrix()
    });

    chart.configure_axes()
        .light_grid_style(BLACK.mix(0.15))
        .max_light_lines(3)
        .draw()?;

    chart.draw_series(
        LineSeries::new(
            bleh.y_out().iter().map(|x| (x[0], x[1], x[2])),
            &BLACK
        ))?
        .label("Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    chart.configure_series_labels().border_style(BLACK).draw()?;
    area.present().expect("Can't write to file");
    Ok(())
}
