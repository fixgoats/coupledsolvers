use num::Complex;
use rand::Rng;
use plotters::prelude::*;

type C64 = Complex<f64>;
// type C32 = Complex<f32>;

const C0: C64 = C64{re: 0., im: 0.};
const CI: C64 = C64{re: 0., im: 1.};
const NSITES: usize = 1;
const NSTEPS: usize = 20000;
const DT: f64 = 0.005;
const GAMMA: f64 = 0.1;
const GAMMA_LP: f64 = 0.012;
const ALPHA: f64 = 0.0004;
const P: f64 = 10.;
const R: f64 = 0.016;
const G: f64 = 0.002;
const OMEGA: f64 = 0.1;
//const D: [usize; NSITES*NSITES] = [0, 12, 20, 12, 0, 8, 20, 12, 0];
//const J: [f64; NSITES*NSITES] = [0., 0.1, 0.01,  0.1, 0., 0.1, 0.01, 0.1, 0.];
const D: [usize; 1] = [0];
const J: [f64; 1] = [0.];
//const BETA: [f64; NSITES*NSITES] = [0., 1., 1.,  1., 0., 1., 1., 1., 0.];
const BETA: [f64; 1] = [0.];
type PsiState = [C64; NSITES];
type XState = [f64; NSITES];
//const PLOTNAME: &str = "semiexacttest.png";


fn fpsi(y: C64, ydelay: &PsiState, x: f64, j: usize) -> C64 {
    /*let mut delayterm = C0;
    for i in 0..NSITES {
        delayterm += J[j*NSITES + i] * (CI*BETA[j*NSITES + i]).exp() * ydelay[i];
    }*/
    return C64{re: 0.5 * (R * x - 0.2), im: -(OMEGA + G * x + ALPHA * y.norm_sqr())} * y;
        /*+ delayterm*/;
}

fn fx(x: f64, y: C64) -> f64 {
    return -(GAMMA + ALPHA * y.norm_sqr()) * x + P;
}

#[allow(dead_code)]
fn init_psi() -> PsiState {
    let mut rng = rand::thread_rng();
    return [(); NSITES].map(|_| C64{re: rng.gen_range(0.0..0.02), im: rng.gen_range(0.0..0.2)});
}

fn rksol() -> Result<(), Box<dyn std::error::Error>> {
    let mut psis = Vec::<PsiState>::with_capacity(NSTEPS);
    let mut xs = Vec::<XState>::with_capacity(NSTEPS);
    //psis.push(init_psi());
    psis.push([0.1*CI; NSITES]);
    xs.push([0.1; NSITES]);

    /*let mut k1psi = [C0; NSITES];
    let mut k2psi = [C0; NSITES];
    let mut k3psi = [C0; NSITES];
    let mut k4psi = [C0; NSITES];
    let mut k1x = [0.; NSITES];
    let mut k2x = [0.; NSITES];
    let mut k3x = [0.; NSITES];
    let mut k4x = [0.; NSITES];*/

    for i in 1..NSTEPS {
        let mut yi = [C0; NSITES];
        let mut xi = [0.; NSITES];
        for j in 0..NSITES {
            let y = psis[i-1][j];
            let x = xs[i-1][j];
            let mut ydelay = [C0; NSITES];
            for k in 0..NSITES {
                let d = D[j*NSITES+k];
                if i <= d || k == j {continue}
                else {ydelay[k] = psis[i-d-1][k];}
            }
            let k1 = fpsi(y, &ydelay, x, j);
            let k2 = fpsi(y + 0.5 * DT * k1, &ydelay, x, j);
            let k3 = fpsi(y + 0.5 * DT * k2, &ydelay, x, j);
            let k4 = fpsi(y + DT * k3, &ydelay, x, j);
            yi[j] = y + (DT / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
            let k1 = fx(x, y);
            let k2 = fx(x + 0.5 * DT * k1, y);
            let k3 = fx(x + 0.5 * DT * k2, y);
            let k4 = fx(x + DT * k3, y);
            xi[j] = x + (DT / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
        }
        psis.push(yi);
        xs.push(xi);
    }
    

    let root_area = BitMapBackend::new("rktest.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let root_area = root_area.titled("Image Title", ("sans-serif", 60))?;

    let (upper, lower) = root_area.split_vertically(540);

    let mut cc = ChartBuilder::on(&upper)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("X", ("sans-serif", 40))
        .build_cartesian_2d(0.0..DT*NSTEPS as f64, -0.1..150.0)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(
            (0..NSTEPS).map(|i| (i as f64 * DT, xs[i][0])), &RED))?
        .label("emmmm")
        .legend(|(x, y)| PathElement::new(vec![(x,y), (x+20, y)], RED));

    let mut cc = ChartBuilder::on(&lower)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("|psi|^2", ("sans-serif", 40))
        .build_cartesian_2d(0.0..DT*NSTEPS as f64, -0.1..20000.0)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(
            (0..NSTEPS).map(|i| (i as f64 * DT, psis[i][0].norm_sqr())), &RED))?
        .label("emmmm")
        .legend(|(x, y)| PathElement::new(vec![(x,y), (x+20, y)], RED));

    root_area.present().expect("úps");
    println!("Made graph rktest.png");
    Ok(())
}

fn semiexact() -> Result<(), Box<dyn std::error::Error>> {
    let mut psis = Vec::<C64>::with_capacity(NSTEPS);
    let mut xs = Vec::<f64>::with_capacity(NSTEPS);
    psis.push(C64{re: 0.0, im: 0.1});
    xs.push(0.1);
    for i in 1..NSTEPS {
        let yprev = psis[i-1];
        let xprev = xs[i-1];
        xs.push(xprev * (-(GAMMA + R * yprev.norm_sqr())*DT).exp() + P * DT);
        psis.push(yprev * (C64{re: 0.5 * R * xs[i], im: -(OMEGA + G * xs[i] + ALPHA * yprev.norm_sqr())}*DT).exp());
    }

    let root_area = BitMapBackend::new("semiexacttest.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let root_area = root_area.titled("Image Title", ("sans-serif", 60))?;

    let (upper, lower) = root_area.split_vertically(540);

    let mut cc = ChartBuilder::on(&upper)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("X", ("sans-serif", 40))
        .build_cartesian_2d(0.0..DT*NSTEPS as f64, -0.1..150.0)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(
            (0..NSTEPS).map(|i| (i as f64 * DT, xs[i])), &RED))?
        .label("emmmm")
        .legend(|(x, y)| PathElement::new(vec![(x,y), (x+20, y)], RED));

    let mut cc = ChartBuilder::on(&lower)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("|psi|^2", ("sans-serif", 40))
        .build_cartesian_2d(0.0..DT*NSTEPS as f64, -0.1..2000.0)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(
            (0..NSTEPS).map(|i| (i as f64 * DT, psis[i].norm_sqr())), &RED))?
        .label("emmmm")
        .legend(|(x, y)| PathElement::new(vec![(x,y), (x+20, y)], RED));

    root_area.present().expect("úps");
    println!("Made graph semiexacttest.png");
    Ok(())
}

fn main() {
    semiexact().ok();
    rksol().ok();
}
