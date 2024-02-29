#[cfg(test)]
mod test {
use plotters::prelude::*;

const DT: f64 = 0.003;
const STEPS: usize = 1000;

fn lerp<T>(a: T, b: T, t: f64) -> T
where T: std::ops::Mul<f64, Output = T> + std::ops::Add<Output = T>
{
    return b * (t / DT) + a * (1. - t / DT);
}

fn fom(omega1: f64, omega2: f64, u: f64) -> f64 {
    return (omega2 - omega1) / u;
}

fn rksol() {
    let mut oms = [0.; STEPS];
    for i in 0..333 {
        oms[i] = 1. / (i as f64 * DT + 1.);
    }
    let t_delay = 1.;
    let d_idx = (t_delay / DT) as usize;
    let idx_t = (d_id + 1) as f64 * DT - t_delay;
    for i in 0..667 {
        let y = oms[333+i];
        let u = (333 + i) as f64 * DT;
        let ydelay1 = oms[333 - d_idx - 1];
        let ydelay2 = oms[333 - d_idx];
        let ydelay = lerp(ydelay1, ydelay2, idx_t);
        let k1 = fom(y, ydelay, u);
        let k2 = fom(y + 0.5 * DT * k1, ydelay, u);
        let k3 = fom(y + 0.5 * DT * k2, ydelay, u);
        let k4 = fom(y + DT * k3, ydelay, u);
        oms[333+i] = y + (DT / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
    }
}
}
