use num::Complex;
use rand::Rng;
use plotters::prelude::*;
use scilib::math::bessel::h1_nu;

type C64 = Complex<f64>;
// type C32 = Complex<f32>;

const C0: C64 = C64{re: 0., im: 0.};
const CI: C64 = C64{re: 0., im: 1.};
const NSITES: usize = 3;
const NSTEPS: usize = 40000;
const DT: f64 = 0.005; // ps
const GAMMA: f64 = 0.1; // ps^{-1}
const GAMMA_LP: f64 = 0.2; // ps^{-1}
const ALPHA: f64 = 0.0004; // ps^{-1}
const P: f64 = 5.; // ps^{-1}
const R: f64 = 0.016; // ps^{-1}
const G: f64 = 0.002; // ps^{-1}
const OMEGA: f64 = 1.22; // ps^{-1}
const D: [f64; NSITES*NSITES] = [0., 20.0, 20.0, 20.0, 0., 20.0, 20.0, 20.0, 0.];
const K0: f64 = 1.62; // µm^{-1}
const KAPPA0: f64 = 0.013; // µm^{-1}
const HBAR: f64 = 0.6582119569; // meV ps
const M: f64 = 0.32; // meV ps^{2} µm^{-2}
const V: f64 = HBAR * K0 / M;
const J0: f64 = 0.01;
const BETA: [f64; NSITES*NSITES] = [0., 1., 1., 1., 0., 1., 1., 1., 0.];
type PsiState = [C64; NSITES];
type XState = [f64; NSITES];

fn lerp<T>(a: T, b: T, r: f64) -> T
where T: std::ops::Mul<f64, Output = T> + std::ops::Add<Output = T>
{
    return b * r + a * (1. - r);
}

fn sym0traceidx<T>(i: usize, j: usize, a: &[T; NSITES*NSITES/2 - 1]) -> T
where T: num::traits::Zero + Copy
{
    if i < j {return a[i*NSITES + j];}
    if i > j {return a[j*NSITES + i];}
    return T::zero();
}

#[allow(unused_variables)]
fn fpsi(y: C64, ydelay: &PsiState, js: &[f64; NSITES*NSITES], x: f64, j: usize) -> C64 {
    let mut delayterm = C0;
    for i in 0..NSITES {
        delayterm += js[j*NSITES + i] * (CI*BETA[j*NSITES + i]).exp() * ydelay[i];
    }
    return C64{re: 0.5 * (R * x - GAMMA_LP), im: -(OMEGA + G * x + ALPHA * y.norm_sqr())} * y
        + delayterm;
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
    psis.push(init_psi());
    xs.push([0.1; NSITES]);
    let t_delay: [f64; NSITES*NSITES] = D.map(|x| x / V); // travel times between sites, time delay
    let d_idx = t_delay.map(|x| (x / DT).ceil() as usize); // next highest usize corresponding to time
                                                    // delay
    let mut idx_t = [0.; NSITES*NSITES]; // the time difference between the actual time delay and
                                         // the index approximated time delay
    for i in 0..NSITES*NSITES {
        idx_t[i] = d_idx[i] as f64 * DT - t_delay[i];
    }
    let d_idx2 = t_delay.map(|x| (x / DT + 0.5).ceil() as usize);
    let mut idx_t2 = [0.; NSITES*NSITES]; // difference between actual time delay and index
                                          // approximated time delay at intermediate step.
    for i in 0..NSITES*NSITES {
        idx_t2[i] = d_idx2[i] as f64 * DT - t_delay[i] + 0.5 * DT;
    }
    let js = D.map(|d| if d != 0. {h1_nu(0., C64{re:K0,im:KAPPA0}).norm() * J0} else {0.});

    for i in 1..NSTEPS {
        let mut yi = [C0; NSITES];
        let mut xi = [0.; NSITES];
        for j in 0..NSITES {
            let y = psis[i-1][j];
            let x = xs[i-1][j];
            let mut ydelay = [C0; NSITES];
            for k in 0..NSITES {
                let d = D[j*NSITES+k];
                if (i as f64) * DT <= d || k == j {continue} // we don't want to
                                                             // index out of bounds
                                                             // or include the same condensate
                let d_idxjk = d_idx[j*NSITES+k];
                let y1 = psis[i-1-d_idxjk][k];
                let y2 = psis[i-d_idxjk][k];
                ydelay[k] = lerp(y1, y2, idx_t[j*NSITES+k]/DT); // interpolate between last recorded
                                                             // value before time delay and next
                                                             // recorded value after time delay to
                                                             // approximate value at time delay
            }
            let k1psi = fpsi(y, &ydelay, &js, x, j);
            let k1x = fx(x, y);
            for k in 0..NSITES {
                let d = D[j*NSITES+k];
                if (i as f64) * DT <= d || k == j {continue} // We've already checked that the index isn't out of bounds,
                                     // now we are only checking that we're not including the same
                                     // condensate
                let d_idxjk = d_idx2[j*NSITES+k];
                let y1 = psis[i-1-d_idxjk][k];
                let y2 = psis[i-d_idxjk][k];
                ydelay[k] = lerp(y1, y2, idx_t2[j*NSITES+k]/DT); 
            }
            let k2psi = fpsi(y + 0.5 * DT * k1psi, &ydelay, &js, x + 0.5 * DT * k1x, j);
            let k2x = fx(x + 0.5 * DT * k1x, y + 0.5 * DT * k1psi);
            let k3psi = fpsi(y + 0.5 * DT * k2psi, &ydelay, &js, x + 0.5 * DT * k2x, j);
            let k3x = fx(x + 0.5 * DT * k2x, y + 0.5 * DT * k3psi);
            for k in 0..NSITES {
                let d = D[j*NSITES+k];
                if (i as f64) * DT <= d || k == j {continue} // We've already checked that the index isn't out of bounds,
                                     // now we are only checking that we're not including the same
                                     // condensate
                let d_idxjk = d_idx[j*NSITES+k] - 1;
                let y1 = psis[i-1-d_idxjk][k];
                let y2 = psis[i-d_idxjk][k];
                ydelay[k] = lerp(y1, y2, idx_t[j*NSITES+k]/DT); 
            }
            let k4psi = fpsi(y + DT * k3psi, &ydelay, &js, x + 0.5 * DT * k3x, j);
            let k4x = fx(x + DT * k3x, y + DT * k3psi);
            yi[j] = y + (DT / 6.) * (k1psi + 2. * k2psi + 2. * k3psi + k4psi);
            xi[j] = x + (DT / 6.) * (k1x + 2. * k2x + 2. * k3x + k4x);
        }
        psis.push(yi);
        xs.push(xi);
    }

    let root_area = BitMapBackend::new("rktest.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let root_area = root_area.titled("Prufa", ("sans-serif", 60))?;

    let ymax = psis.iter().map(|x| x.iter().map(|v| v.norm_sqr()).reduce(f64::max).unwrap()).reduce(f64::max).unwrap();

    let mut cc = ChartBuilder::on(&root_area)
        .margin_top(0)
        .margin_left(0)
        .margin_right(30)
        .margin_bottom(30)
        .x_label_area_size(70)
        .y_label_area_size(90)
        .build_cartesian_2d(0.0..DT*NSTEPS as f64, -0.1..(1.05*ymax))?;

    cc.configure_mesh()
        .x_label_formatter(&|x| format!("{:.1}", *x))
        .y_label_formatter(&|y| format!("{:.1}", *y))
        .x_desc("t [ps]")
        .y_desc("|ψ|^2")
        .x_labels(10)
        .y_labels(10)
        .axis_desc_style(("sans-serif", 40))
        .label_style(("sans-serif", 30))
        .draw()?;

    cc.draw_series(LineSeries::new(
            (0..NSTEPS).map(|i| (i as f64 * DT, psis[i][0].norm_sqr())), RGBColor(255, 0, 0)))?
        .label("Site 1")
        .legend(|(x, y)| PathElement::new(vec![(x,y), (x+40, y)], ShapeStyle{color: RGBAColor(255, 0, 0, 1.0),
                                                                             filled: true,
                                                                             stroke_width: 2}));


    cc.draw_series(LineSeries::new(
            (0..NSTEPS).map(|i| (i as f64 * DT, psis[i][1].norm_sqr())), RGBColor(0,0,255)))?
        .label("Site 2")
        .legend(|(x, y)| PathElement::new(vec![(x,y), (x+40, y)], ShapeStyle{color: RGBAColor(0, 0, 255, 1.0),
                                                                             filled: true,
                                                                             stroke_width: 2}));

    cc.configure_series_labels().legend_area_size(75).label_font(("sans-serif", 30)).border_style(BLACK).draw()?;
    root_area.present().expect("úps");
    println!("Made graph rktest.png");
    Ok(())
}

/*fn semiexact() -> Result<(), Box<dyn std::error::Error>> {
    let mut psis = Vec::<C64>::with_capacity(NSTEPS);
    let mut xs = Vec::<f64>::with_capacity(NSTEPS);
    psis.push(C64{re: 0.0, im: 0.1});
    xs.push(0.1);
    for i in 1..NSTEPS {
        let yprev = psis[i-1];
        let xprev = xs[i-1];
        xs.push(xprev * (-(GAMMA + R * yprev.norm_sqr())*DT).exp() + P * DT);
        psis.push(yprev * (C64{re: 0.5 * (R * xprev - GAMMA_LP), im: -(OMEGA + G * xprev + ALPHA * yprev.norm_sqr())}*DT).exp());
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
}*/

fn main() {
    //semiexact().ok();
    rksol().ok();
    // buchstabtest().ok();
}

fn fom(omega1: f64, omega2: f64, u: f64) -> f64 {
    return (omega2 - omega1) / u;
}

fn buchstabtest() -> Result<(), Box<dyn std::error::Error>> {
    let mut oms = [0.; 3000];
    let dt = 0.001;
    for i in 0..1001 {
        oms[i] = 1. / (i as f64 * dt + 1.);
    }
    let t_delay = 1.;
    let d_idx = (t_delay / dt).ceil() as usize;
    let d_idx2 = (t_delay / dt - 0.5).ceil() as usize;
    let idx_t = d_idx as f64 * dt - t_delay; // amount of time between approximate index
                                                   // time and actual delay time, for
                                                   // interpolation. It would be possible to divide
                                                   // through by dt here for a tiny bit of extra
                                                   // performance, but that would make it even less
                                                   // clear what's happening here and the optimizer
                                                   // probably takes care of it anyway, and even if
                                                   // it doesn't the difference is probably
                                                   // negligible.
    println!("{}", idx_t);
    let idx_t2 = (d_idx2 as f64 + 0.5) * dt - t_delay;
    for i in 1000..2999 {
        let y = oms[i];
        let u = i as f64 * dt + 1.;
        let mut ydelay1 = oms[i - d_idx];
        let mut ydelay2 = oms[i + 1 - d_idx];
        let mut ydelay = lerp(ydelay1, ydelay2, idx_t/dt);
        let k1 = fom(y, ydelay, u);
        ydelay1 = oms[i  - d_idx2];
        ydelay2 = oms[i + 1 - d_idx2];
        ydelay = lerp(ydelay1, ydelay2, idx_t2 / dt);
        let k2 = fom(y + 0.5 * dt * k1, ydelay, u + 0.5 * dt);
        let k3 = fom(y + 0.5 * dt * k2, ydelay, u + 0.5 * dt);
        ydelay1 = oms[i + 1 - d_idx];
        ydelay2 = oms[i + 2 - d_idx];
        ydelay = lerp(ydelay1, ydelay2, idx_t / dt);
        let k4 = fom(y + dt * k3, ydelay, u + dt);
        oms[i + 1] = y + (dt / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
    }
    /*for i in 1000..2999 {
        let y = oms[i];
        let ydelay1 = oms[i - d_idx];
        let ydelay2 = oms[i + 1 - d_idx];
        let k1 = fom(y, ydelay1, i as f64 * dt + 1.);
        let k2 = fom(y + dt*k1, ydelay2, (i+1) as f64 * dt + 1.);
        oms[i+1] = y + 0.5 * dt * (k1 + k2);
    }*/
    println!("{}", oms[1001]);

    let root_area = BitMapBackend::new("buchstabtest.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let root_area = root_area.titled("|psi|^2", ("sans-serif", 60))?;

    let mut cc = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(1.0..(dt*3000.+1.) as f64, 0.4..1.1)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(
            (0..3000).map(|i| (i as f64 * dt + 1., oms[i])), &RED))?
        .label("emmmm")
        .legend(|(x, y)| PathElement::new(vec![(x,y), (x+20, y)], RED));

    root_area.present().expect("úps");
    Ok(())
}
