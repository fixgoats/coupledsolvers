use num::Complex;
use rand::Rng;
use plotters::prelude::*;
use scilib::math::bessel::h1_nu;
use serde::{Serialize, Deserialize};
use clap::Parser;
use std::fs::{File,OpenOptions};

type C64 = Complex<f64>;
// type C32 = Complex<f32>;

const C0: C64 = C64{re: 0., im: 0.};
const C1: C64 = C64{re: 1., im: 0.};
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

#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    #[arg(short, long, default_value_t = false)]
    cached: bool,
}

fn lerp<T>(a: T, b: T, r: f64) -> T
where T: std::ops::Mul<f64, Output = T> + std::ops::Add<Output = T>
{
    b * r + a * (1. - r)
}

fn sym0traceidx<T>(i: usize, j: usize, a: &[T; NSITES*NSITES/2 - 1]) -> T
where T: num::traits::Zero + Copy
{
    if i < j {return a[i*NSITES + j];}
    if i > j {return a[j*NSITES + i];}
    T::zero()
}

struct DelayPsi {
    psi: C64,
    j: f64,
    beta: C64
}

#[allow(unused_variables)]
fn fpsi(y: C64, ydelay: &Vec<DelayPsi>, x: f64) -> C64 {
    let mut delayterm = C0;
    for y in ydelay {
        delayterm += y.j * y.beta * y.psi;
    }
    C64{re: 0.5 * (R * x - GAMMA_LP), im: -(OMEGA + G * x + ALPHA * y.norm_sqr())} * y
        + delayterm
}

fn fx(x: f64, y: C64) -> f64 {
    -(GAMMA + ALPHA * y.norm_sqr()) * x + P
}

#[allow(dead_code)]
fn init_psi(n: usize) -> Vec<C64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| C64{re: rng.gen_range(0.0..0.02), im: rng.gen_range(0.0..0.2)}).collect()
}

fn p3filter(scale: f64, r: f64) -> Vec<C64> {
    let mut points = Vec::<C64>::with_capacity(191);
    for p in P3 {
        if (p*scale).norm_sqr() < r*r*scale*scale {
            points.push(p*scale);
        }
    }
    points.shrink_to_fit();
    points
}

#[derive(Serialize, Deserialize)]
struct Neighbour {
    idx: usize,
    d_idx: usize,
    dt: f64,
    d2_idx: usize,
    dt2: f64,
    j: f64,
    beta: C64
}

fn find_neighbours(points: &Vec<C64>, r: f64) -> Vec<Vec<Neighbour>> {
    let mut neighbours = Vec::<Vec<Neighbour>>::with_capacity(points.len());
    for i in 0..points.len() {
        neighbours.push(Vec::<Neighbour>::with_capacity(6));
        for j in 0..points.len() {
            if i == j {continue;}
            let d = (points[i] - points[j]).norm();
            if d < r {
                let delay = d / V;
                let d_idx = (delay / DT).ceil() as usize;
                let dt = (d_idx as f64) * DT - delay;
                let d2_idx = (delay / DT - 0.5).ceil() as usize;
                let dt2 = (d2_idx as f64) * DT - delay + 0.5 * DT;
                let jij = J0*h1_nu(0., (K0*C1 + 0.001*CI)* delay).norm();
                let beta = C0;
                neighbours[i].push(Neighbour{
                    idx: j,
                    d_idx,
                    dt,
                    d2_idx,
                    dt2,
                    j: jij,
                    beta
                });
            }
        }
        neighbours[i].shrink_to_fit();
    }
    neighbours
}

fn plotp3() -> Result<(), Box<dyn std::error::Error>> {
    let points = p3filter(20., 3.9);
    let xmin = points.iter().map(|x| x.re).reduce(f64::min).unwrap();
    let xmax = points.iter().map(|x| x.re).reduce(f64::max).unwrap();
    let ymin = points.iter().map(|x| x.im).reduce(f64::min).unwrap();
    let ymax = points.iter().map(|x| x.im).reduce(f64::max).unwrap();
    let root_area = BitMapBackend::new("p3.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let root_area = root_area.titled("Prufa", ("sans-serif", 60))?;

    let mut cc = ChartBuilder::on(&root_area)
        .margin_top(70)
        .margin_left(70)
        .margin_right(70)
        .margin_bottom(70)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

    cc.configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .axis_desc_style(("sans-serif", 40))
        .label_style(("sans-serif", 30))
        .draw()?;

    cc.draw_series(
            points.iter().map(|x| Circle::new((x.re, x.im), 4., RGBColor(255, 0, 0))))?;
    println!("{}", points.len());
    Ok(())
}

fn neighbourtest() {
    let points = p3filter(20., 3.9);
    let neighbours = find_neighbours(&points, 20.);
    println!("length: {}, should be {}\nlength of random index {}", neighbours.len(), points.len(), neighbours[6].len())
}

// Besides cleanliness, this is pulled into a separate function so that the serialized byte vector is freed
// when it goes out of scope, thus saving memory
fn save_neighbours(v: &Vec<u8>) -> std::io::Result<()> {
    let serialized = serde_binary::to_vec(&v, serde_binary::Endian::Little);
    let mut file = OpenOptions::new()
        .create(true).write(true).open("neighbours.bin")?;
    file.write_all(&serialized); 
    Ok(())
}

fn read_neighbours() -> Vec<Vec<Neighbour>> {
    let bleh = File::open("neighbours.bin").expect("no file found");
    let neighbours = serde_binary::from_vec::<Vec<Vec<Neighbour>>>(&bleh).unwrap();
}

fn rksol() -> Result<(), Box<dyn std::error::Error>> {
    let points = p3filter(20., 3.9);
    let nsites = points.len();
    let mut psis = Vec::<Vec<C64>>::with_capacity(NSTEPS);
    let mut xs = Vec::<Vec<f64>>::with_capacity(NSTEPS);
    psis.push(init_psi(nsites));
    xs.push(vec![0.01; nsites]);
    let args = Args::parse();
    let neighbours = match args.cache {
        true => 
    }
    let neighbours = find_neighbours(&points, 20.);
    println!("found neighbours");
    save_neighbours(&neighbours);


    for i in 1..NSTEPS {
        let mut yi = Vec::with_capacity(nsites);
        let mut xi = Vec::with_capacity(nsites);
        for j in 0..nsites {
            let y = psis[i-1][j];
            let x = xs[i-1][j];
            let mut ydelay = Vec::with_capacity(neighbours[j].len());
            for n in &neighbours[j] {
                if i < n.d_idx {continue;}
                let y1 = psis[i-n.d_idx][n.idx];
                let y2 = psis[i-n.d_idx+1][n.idx];
                ydelay.push(DelayPsi{
                    psi: lerp(y1, y2, n.dt / DT),
                    j: n.j,
                    beta: n.beta
                });
            }
            let k1psi = fpsi(y, &ydelay, x);
            let k1x = fx(x, y);
            ydelay.clear();
            for n in &neighbours[j] {
                if i < n.d_idx {continue;}
                let y1 = psis[i-n.d2_idx][n.idx];
                let y2 = psis[i-n.d2_idx+1][n.idx];
                ydelay.push(DelayPsi{
                    psi: lerp(y1, y2, n.dt2 / DT),
                    j: n.j,
                    beta: n.beta
                });
            }
            let k2psi = fpsi(y + 0.5 * DT * k1psi, &ydelay, x + 0.5 * DT * k1x);
            let k2x = fx(x + 0.5 * DT * k1x, y + 0.5 * DT * k1psi);
            let k3psi = fpsi(y + 0.5 * DT * k2psi, &ydelay, x + 0.5 * DT * k2x);
            let k3x = fx(x + 0.5 * DT * k2x, y + 0.5 * DT * k3psi);
            for n in &neighbours[j] {
                if i < n.d_idx {continue;}
                let y1 = psis[i-n.d_idx+1][n.idx];
                let y2 = psis[i-n.d_idx+2][n.idx];
                ydelay.push(DelayPsi{
                    psi: lerp(y1, y2, n.dt / DT),
                    j: n.j,
                    beta: n.beta
                });
            }
            let k4psi = fpsi(y + DT * k3psi, &ydelay, x + 0.5 * DT * k3x);
            let k4x = fx(x + DT * k3x, y + DT * k3psi);
            yi.push(y + (DT / 6.) * (k1psi + 2. * k2psi + 2. * k3psi + k4psi));
            xi.push(x + (DT / 6.) * (k1x + 2. * k2x + 2. * k3x + k4x));
        }
        psis.push(yi);
        xs.push(xi);
    }

    let root_area = BitMapBackend::new("rktestpsisq.png", (1920, 1080)).into_drawing_area();
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
    println!("Made graph rktestpsisq.png");

    let root_area = BitMapBackend::new("rktestargs.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let root_area = root_area.titled("Prufa", ("sans-serif", 60))?;

    let args: Vec<f64> = psis.iter().map(|x| (x[0].conj()*x[1]).arg()).collect();

    let ymin = args.iter().map(|x| *x).reduce(f64::min).unwrap();
    let ymax = args.iter().map(|x| *x).reduce(f64::max).unwrap();

    let mut cc = ChartBuilder::on(&root_area)
        .margin_top(0)
        .margin_left(0)
        .margin_right(30)
        .margin_bottom(30)
        .x_label_area_size(70)
        .y_label_area_size(90)
        .build_cartesian_2d(0.0..DT*NSTEPS as f64, (1.05*ymin)..(1.05*ymax))?;

    cc.configure_mesh()
        .x_label_formatter(&|x| format!("{:.1}", *x))
        .y_label_formatter(&|y| format!("{:.1}", *y))
        .x_desc("t [ps]")
        .y_desc("")
        .x_labels(10)
        .y_labels(10)
        .axis_desc_style(("sans-serif", 40))
        .label_style(("sans-serif", 30))
        .draw()?;

    cc.draw_series(LineSeries::new(
            (0..NSTEPS).map(|i| (i as f64 * DT, args[i])), RGBColor(255, 0, 0)))?
        .label("ψ_1^*ψ_2")
        .legend(|(x, y)| PathElement::new(vec![(x,y), (x+40, y)], ShapeStyle{color: RGBAColor(255, 0, 0, 1.0),
                                                                             filled: true,
                                                                             stroke_width: 2}));


    cc.configure_series_labels().legend_area_size(75).label_font(("sans-serif", 30)).border_style(BLACK).draw()?;
    root_area.present().expect("úps");
    println!("Made graph rktestarg.png");

    Ok(())
}


fn main() {
    //semiexact().ok();
    rksol().ok();
    // plotp3().ok();
    // neighbourtest();
    // buchstabtest().o
}

fn fom(omega1: f64, omega2: f64, u: f64) -> f64 {
    (omega2 - omega1) / u
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

const P3: [C64; 191] = [ 
    C64{re: 3.45491503e+00, im: 1.12256994e+00},
    C64{re: 3.68033989e+00, im: 4.28783563e-01},
    C64{re: 4.04508497e+00, im: 6.93786379e-01},
    C64{re: 3.09016994e+00, im: 0.00000000e+00},
    C64{re: 4.27050983e+00, im: 0.00000000e+00},
    C64{re: 5.00000000e+00, im: 0.00000000e+00},
    C64{re: 3.68033989e+00, im: 1.81635632e+00},
    C64{re: 4.40983006e+00, im: 1.81635632e+00},
    C64{re: 4.27050983e+00, im: 2.24513988e+00},
    C64{re: 4.04508497e+00, im: 2.93892626e+00},
    C64{re: 4.77457514e+00, im: 6.93786379e-01},
    C64{re: 4.18440520e+00, im: 1.12256994e+00},
    C64{re: 2.13525492e+00, im: 6.93786379e-01},
    C64{re: 1.54508497e+00, im: 1.12256994e+00},
    C64{re: 1.90983006e+00, im: 0.00000000e+00},
    C64{re: 2.36067977e+00, im: 0.00000000e+00},
    C64{re: 7.29490169e-01, im: 0.00000000e+00},
    C64{re: 0.00000000e+00, im: 0.00000000e+00},
    C64{re: 9.54915028e-01, im: 6.93786379e-01},
    C64{re: 1.31966011e+00, im: 4.28783563e-01},
    C64{re: 3.09016994e+00, im: 2.24513988e+00},
    C64{re: 3.45491503e+00, im: 2.51014270e+00},
    C64{re: 2.13525492e+00, im: 1.55135350e+00},
    C64{re: 2.72542486e+00, im: 1.12256994e+00},
    C64{re: 2.86474508e+00, im: 1.55135350e+00},
    C64{re: 2.86474508e+00, im: 6.93786379e-01},
    C64{re: 2.13525492e+00, im: 2.93892626e+00},
    C64{re: 1.54508497e+00, im: 3.36770982e+00},
    C64{re: 1.90983006e+00, im: 3.63271264e+00},
    C64{re: 9.54915028e-01, im: 2.93892626e+00},
    C64{re: 1.31966011e+00, im: 4.06149620e+00},
    C64{re: 1.54508497e+00, im: 4.75528258e+00},
    C64{re: 2.86474508e+00, im: 2.93892626e+00},
    C64{re: 3.09016994e+00, im: 3.63271264e+00},
    C64{re: 3.45491503e+00, im: 3.36770982e+00},
    C64{re: 2.13525492e+00, im: 4.32649902e+00},
    C64{re: 2.36067977e+00, im: 3.63271264e+00},
    C64{re: 1.31966011e+00, im: 1.81635632e+00},
    C64{re: 5.90169944e-01, im: 1.81635632e+00},
    C64{re: 7.29490169e-01, im: 2.24513988e+00},
    C64{re: 2.25424859e-01, im: 6.93786379e-01},
    C64{re: 8.15594803e-01, im: 1.12256994e+00},
    C64{re: 1.90983006e+00, im: 2.24513988e+00},
    C64{re: 2.36067977e+00, im: 2.24513988e+00},
    C64{re: 1.54508497e+00, im: 2.51014270e+00},
    C64{re: 2.22044605e-16, im: 3.63271264e+00},
    C64{re: 7.29490169e-01, im: 3.63271264e+00},
    C64{re: 5.90169944e-01, im: 4.06149620e+00},
    C64{re: -5.90169944e-01, im: 4.06149620e+00},
    C64{re: -3.64745084e-01, im: 4.75528258e+00},
    C64{re: -8.15594803e-01, im: 4.75528258e+00},
    C64{re: -1.54508497e+00, im: 4.75528258e+00},
    C64{re: 8.15594803e-01 , im: 4.75528258e+00},
    C64{re: 2.25424859e-01 , im: 4.32649902e+00},
    C64{re: 0.00000000e+00 , im: 2.24513988e+00},
    C64{re: -5.90169944e-01, im: 1.81635632e+00},
    C64{re: -3.64745084e-01, im: 1.12256994e+00},
    C64{re: 1.11022302e-16 , im: 1.38757276e+00},
    C64{re: -1.18033989e+00, im: 3.63271264e+00},
    C64{re: -1.31966011e+00, im: 4.06149620e+00},
    C64{re: -8.15594803e-01, im: 2.51014270e+00},
    C64{re: -2.25424859e-01, im: 2.93892626e+00},
    C64{re: -5.90169944e-01, im: 3.20392908e+00},
    C64{re: 2.25424859e-01 , im: 2.93892626e+00},
    C64{re: -2.13525492e+00, im: 2.93892626e+00},
    C64{re: -2.72542486e+00, im: 2.51014270e+00},
    C64{re: -2.86474508e+00, im: 2.93892626e+00},
    C64{re: -2.50000000e+00, im: 1.81635632e+00},
    C64{re: -3.45491503e+00, im: 2.51014270e+00},
    C64{re: -4.04508497e+00, im: 2.93892626e+00},
    C64{re: -1.90983006e+00, im: 3.63271264e+00},
    C64{re: -2.50000000e+00, im: 4.06149620e+00},
    C64{re: -2.13525492e+00, im: 4.32649902e+00},
    C64{re: -3.45491503e+00, im: 3.36770982e+00},
    C64{re: -2.72542486e+00, im: 3.36770982e+00},
    C64{re: -1.31966011e+00, im: 1.81635632e+00},
    C64{re: -1.54508497e+00, im: 1.12256994e+00},
    C64{re: -1.90983006e+00, im: 1.38757276e+00},
    C64{re: -5.90169944e-01, im: 4.28783563e-01},
    C64{re: -8.15594803e-01, im: 1.12256994e+00},
    C64{re: -1.54508497e+00, im: 2.51014270e+00},
    C64{re: -1.40576475e+00, im: 2.93892626e+00},
    C64{re: -1.90983006e+00, im: 2.24513988e+00},
    C64{re: -3.45491503e+00, im: 1.12256994e+00},
    C64{re: -3.22949017e+00, im: 1.81635632e+00},
    C64{re: -3.68033989e+00, im: 1.81635632e+00},
    C64{re: -4.04508497e+00, im: 6.93786379e-01},
    C64{re: -4.63525492e+00, im: 1.12256994e+00},
    C64{re: -4.77457514e+00, im: 6.93786379e-01},
    C64{re: -5.00000000e+00, im: 6.12323400e-16},
    C64{re: -4.27050983e+00, im: 2.24513988e+00},
    C64{re: -4.04508497e+00, im: 1.55135350e+00},
    C64{re: -2.13525492e+00, im: 6.93786379e-01},
    C64{re: -1.90983006e+00, im: 2.33886727e-16},
    C64{re: -1.18033989e+00, im: 1.44549947e-16},
    C64{re: -1.31966011e+00, im: 4.28783563e-01},
    C64{re: -3.81966011e+00, im: 4.67773453e-16},
    C64{re: -4.27050983e+00, im: 5.22986620e-16},
    C64{re: -2.63932023e+00, im: 3.23223507e-16},
    C64{re: -2.86474508e+00, im: 6.93786379e-01},
    C64{re: -3.22949017e+00, im: 4.28783563e-01},
    C64{re: -2.72542486e+00, im: 1.12256994e+00},
    C64{re: -3.45491503e+00, im: -1.12256994e+00},
    C64{re: -3.22949017e+00, im: -1.81635632e+00},
    C64{re: -3.68033989e+00, im: -1.81635632e+00},
    C64{re: -2.50000000e+00, im: -1.81635632e+00},
    C64{re: -3.45491503e+00, im: -2.51014270e+00},
    C64{re: -4.04508497e+00, im: -2.93892626e+00},
    C64{re: -4.04508497e+00, im: -6.93786379e-01},
    C64{re: -4.63525492e+00, im: -1.12256994e+00},
    C64{re: -4.77457514e+00, im: -6.93786379e-01},
    C64{re: -4.27050983e+00, im: -2.24513988e+00},
    C64{re: -4.04508497e+00, im: -1.55135350e+00},
    C64{re: -2.13525492e+00, im: -6.93786379e-01},
    C64{re: -1.54508497e+00, im: -1.12256994e+00},
    C64{re: -1.90983006e+00, im: -1.38757276e+00},
    C64{re: -5.90169944e-01, im: -4.28783563e-01},
    C64{re: -1.31966011e+00, im: -4.28783563e-01},
    C64{re: -2.86474508e+00, im: -6.93786379e-01},
    C64{re: -3.22949017e+00, im: -4.28783563e-01},
    C64{re: -2.72542486e+00, im: -1.12256994e+00},
    C64{re: -2.13525492e+00, im: -2.93892626e+00},
    C64{re: -2.72542486e+00, im: -2.51014270e+00},
    C64{re: -2.86474508e+00, im: -2.93892626e+00},
    C64{re: -1.90983006e+00, im: -3.63271264e+00},
    C64{re: -2.50000000e+00, im: -4.06149620e+00},
    C64{re: -2.13525492e+00, im: -4.32649902e+00},
    C64{re: -1.54508497e+00, im: -4.75528258e+00},
    C64{re: -3.45491503e+00, im: -3.36770982e+00},
    C64{re: -2.72542486e+00, im: -3.36770982e+00},
    C64{re: -1.31966011e+00, im: -1.81635632e+00},
    C64{re: -5.90169944e-01, im: -1.81635632e+00},
    C64{re: -3.64745084e-01, im: -1.12256994e+00},
    C64{re: -8.15594803e-01, im: -1.12256994e+00},
    C64{re: -1.18033989e+00, im: -3.63271264e+00},
    C64{re: -1.31966011e+00, im: -4.06149620e+00},
    C64{re: -8.15594803e-01, im: -2.51014270e+00},
    C64{re: -1.54508497e+00, im: -2.51014270e+00},
    C64{re: -1.40576475e+00, im: -2.93892626e+00},
    C64{re: -1.90983006e+00, im: -2.24513988e+00},
    C64{re: -6.66133815e-16, im: -3.63271264e+00},
    C64{re: 7.29490169e-01 , im: -3.63271264e+00},
    C64{re: 5.90169944e-01 , im: -4.06149620e+00},
    C64{re: 9.54915028e-01 , im: -2.93892626e+00},
    C64{re: 1.31966011e+00 , im: -4.06149620e+00},
    C64{re: 1.54508497e+00 , im: -4.75528258e+00},
    C64{re: -5.90169944e-01, im: -4.06149620e+00},
    C64{re: -3.64745084e-01, im: -4.75528258e+00},
    C64{re: -8.15594803e-01, im: -4.75528258e+00},
    C64{re: 8.15594803e-01 , im: -4.75528258e+00},
    C64{re: 2.25424859e-01 , im: -4.32649902e+00},
    C64{re: -3.33066907e-16, im: -2.24513988e+00},
    C64{re: 5.90169944e-01 , im: -1.81635632e+00},
    C64{re: 7.29490169e-01 , im: -2.24513988e+00},
    C64{re: 2.25424859e-01 , im: -6.93786379e-01},
    C64{re: -2.22044605e-16, im: -1.38757276e+00},
    C64{re: -2.25424859e-01, im: -2.93892626e+00},
    C64{re: -5.90169944e-01, im: -3.20392908e+00},
    C64{re: 2.25424859e-01 , im: -2.93892626e+00},
    C64{re: 2.13525492e+00 , im: -2.93892626e+00},
    C64{re: 1.54508497e+00 , im: -3.36770982e+00},
    C64{re: 1.90983006e+00 , im: -3.63271264e+00},
    C64{re: 2.86474508e+00 , im: -2.93892626e+00},
    C64{re: 3.09016994e+00 , im: -3.63271264e+00},
    C64{re: 3.45491503e+00 , im: -3.36770982e+00},
    C64{re: 4.04508497e+00 , im: -2.93892626e+00},
    C64{re: 2.13525492e+00 , im: -4.32649902e+00},
    C64{re: 2.36067977e+00 , im: -3.63271264e+00},
    C64{re: 1.31966011e+00 , im: -1.81635632e+00},
    C64{re: 1.54508497e+00 , im: -1.12256994e+00},
    C64{re: 9.54915028e-01 , im: -6.93786379e-01},
    C64{re: 8.15594803e-01 , im: -1.12256994e+00},
    C64{re: 3.09016994e+00 , im: -2.24513988e+00},
    C64{re: 3.45491503e+00 , im: -2.51014270e+00},
    C64{re: 2.13525492e+00 , im: -1.55135350e+00},
    C64{re: 1.90983006e+00 , im: -2.24513988e+00},
    C64{re: 2.36067977e+00 , im: -2.24513988e+00},
    C64{re: 1.54508497e+00 , im: -2.51014270e+00},
    C64{re: 3.45491503e+00 , im: -1.12256994e+00},
    C64{re: 3.68033989e+00 , im: -4.28783563e-01},
    C64{re: 4.04508497e+00 , im: -6.93786379e-01},
    C64{re: 3.68033989e+00 , im: -1.81635632e+00},
    C64{re: 4.40983006e+00 , im: -1.81635632e+00},
    C64{re: 4.27050983e+00 , im: -2.24513988e+00},
    C64{re: 4.77457514e+00 , im: -6.93786379e-01},
    C64{re: 4.18440520e+00 , im: -1.12256994e+00},
    C64{re: 2.13525492e+00 , im: -6.93786379e-01},
    C64{re: 1.31966011e+00 , im: -4.28783563e-01},
    C64{re: 2.72542486e+00 , im: -1.12256994e+00},
    C64{re: 2.86474508e+00 , im: -1.55135350e+00},
    C64{re: 2.86474508e+00 , im: -6.93786379e-01}];
