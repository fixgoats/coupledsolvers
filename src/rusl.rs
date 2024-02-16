use num::Complex;
use ndarray::{Array1, Array2, array};

type C64 = Complex<f64>;
type C32 = Complex<f32>;

const C1: C64 = C64{re: 1., im: 0.};
const CI: C64 = C64{re: 0., im: 1.};
const C0: C64 = C64{re: 0., im: 0.};
const EMPTYCARRAY: Array1<C64> = Array1::<C64>::from_vec(vec![]);
const EMPTYRARRAY: Array1<f64> = Array1::<f64>::from_vec(vec![]);

struct System {
    j: Array2<f64>,
    d: Array2<f64>,
    psi: Array1<C64>,
    x: Array1<f64>,
    k1: Array1<C64>,
    k2: Array1<C64>,
    k3: Array1<C64>,
    k4: Array1<C64>,
    xk1: Array1<f64>,
    xk2: Array1<f64>,
    xk3: Array1<f64>,
    xk4: Array1<f64>,
    tmppsi: Array1<C64>,
    tmpx: Array1<f64>,
    alpha: f64,
    omega: f64,
    g: f64,
    r: f64,
    beta: f64,
    v: f64,
    gamma: f64,
    p: f64,
    n: usize
}

impl System {
    fn new(j: Array2<f64>,
           d: Array2<f64>,
           psi: Array1<C64>,
           x: Array1<f64>,
           alpha: f64,
           omega: f64,
           g: f64,
           r: f64,
           beta: f64,
           v: f64,
           gamma: f64,
           p: f64) -> System {
        let n = psi.len();
        if x.len() != n {panic!("psi and x not same length")};
        if j.shape() != [n, n] {panic!("j not right shape")};
        if d.shape() != [n, n] {panic!("d not right shape")};
        let tmppsi = Array1::from(psi);
        let tmpx = Array1::from(x);
        let k1 = Array1::<C64>::zeros(n);
        let k2 = Array1::<C64>::zeros(n);
        let k3 = Array1::<C64>::zeros(n);
        let k4 = Array1::<C64>::zeros(n);
        let xk1 = Array1::<f64>::zeros(n);
        let xk2 = Array1::<f64>::zeros(n);
        let xk3 = Array1::<f64>::zeros(n);
        let xk4 = Array1::<f64>::zeros(n);
        System {
            j,
            d,
            psi,
            x,
            k1,
            k2,
            k3,
            k4,
            xk1,
            xk2,
            xk3,
            xk4,
            tmppsi,
            tmpx,
            alpha,
            omega,
            g,
            r,
            beta,
            v,
            gamma,
            p,
            n
        }
    }
    fn fpsi(&mut self, h: f64, k: &Array1<C64>, output: &mut Array1<C64>) {
        for i in 0..self.n {
            let psi = if k.len() == 0 {self.psi[i]} else {self.psi[i] + h * k[i]};
            let psisqr = psi.norm_sqr();
            output[i] = C64{
                re: -self.r * self.x[i],
                im: self.omega + self.g * self.x[i] + self.alpha * psisqr}
                * psi;
        }
    }

    fn fx(&mut self, h: f64, xk: &Array1<f64>, output: &mut Array1<f64>) {
        for i in 0..self.n {
            let x = if xk.len() == 0 {self.x[i]} else {self.x[i] + h * xk[i]};
            let psisqr = self.psi[i].norm_sqr();
            output[i] = -(self.gamma + self.r * psisqr) * x + self.p;
        }
    }

    fn step(&mut self, dt: f64) {
        self.fpsi(0., &EMPTYCARRAY, &mut self.k1);
        self.fpsi(0.5 * dt, &self.k1, &mut self.k2);
        self.fpsi(0.5 * dt, &self.k2, &mut self.k3);
        self.fpsi(dt, &self.k3, &mut self.k4);
        for i in 0..self.n {
            let inc = self.k1[i] + 2. * self.k2[i] + 2. * self.k3[i] + self.k4[i];
            self.psi[i] += dt * inc / 6.;
        }
        self.fx(0., &EMPTYRARRAY, &mut self.xk1);
        self.fx(0.5 * dt, &self.xk1, &mut self.xk2);
        self.fx(0.5 * dt, &self.xk2, &mut self.xk3);
        self.fx(dt, &self.xk3, &mut self.xk3);
        for i in 0..self.n {
            let inc = self.xk1[i] + 2. * self.xk2[i] + 2. * self.xk3[i] + self.xk4[i];
            self.x[i] += dt * inc / 6.;
        }
    }
}

fn main() {
    let mut condensate = System::new(
        array![[0., 1.],
               [1., 0.]],
        array![
            [0., 0.],
            [0., 0.]
        ],
        array![C0, C0],
        array![0., 0.],
        0.1,
        1.,
        1.,
        1.,
        1.,
        1.,
        0.01,
        1.
    );

    
}
