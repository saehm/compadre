// https://renecutura.eu v0.0.2 Copyright 2019 Rene Cutura
(function (global, factory) {
typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
typeof define === 'function' && define.amd ? define(['exports'], factory) :
(global = global || self, factory(global.druid = global.druid || {}));
}(this, function (exports) { 'use strict';

/**
 * Computes the euclidean distance (l_2) between {@link a} and {@link b}.
 * @memberof module:metrics
 * @alias euclidean
 * @param {Array<Number>} a 
 * @param {Array<Number>} b 
 * @returns {Number} the euclidean distance between {@link a} and {@link b}.  
 */
function euclidean(a, b) {
    return Math.sqrt(euclidean_squared(a, b));
}

/**
 * Numerical stable summation with the Kahan summation algorithm.
 * @memberof module:numerical
 * @alias kahan_sum
 * @param {Array} summands - Array of values to sum up.
 * @returns {number} The sum.
 * @see {@link https://en.wikipedia.org/wiki/Kahan_summation_algorithm}
 */
function kahan_sum(summands) {
    let n = summands.length;
    let sum = 0;
    let compensation = 0;
    let y, t;

    for (let i = 0; i < n; ++i) {
        y = summands[i] - compensation;
        t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    return sum;
}

/**
 * Numerical stable summation with the Neumair summation algorithm.
 * @memberof module:numerical
 * @alias neumair_sum
 * @param {Array} summands - Array of values to sum up.
 * @returns {number} The sum.
 * @see {@link https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements}
 */
function neumair_sum(summands) {
    let n = summands.length;
    let sum = 0;
    let compensation = 0;

    for (let i = 0; i < n; ++i) {
        let summand = summands[i];
        let t = sum + summand;
        if (Math.abs(sum) >= Math.abs(summand)) {
            compensation += (sum - t) + summand;
        } else {
            compensation += (summand - t) + sum;
        }
        sum = t;
    }
    return sum + compensation;
}

/**
 * @module numerical
 */

/**
 * Computes the squared euclidean distance (l_2^2) between {@link a} and {@link b}.
 * @memberof module:metrics
 * @alias euclidean_squared
 * @param {Array<Number>} a 
 * @param {Array<Number>} b 
 * @returns {Number} the squared euclidean distance between {@link a} and {@link b}.  
 */
function euclidean_squared(a, b) {
    if (a.length != b.length) return undefined
    let n = a.length;
    let s = new Array(n);
    for (let i = 0; i < n; ++i) {
        let x = a[i];
        let y = b[i];
        s[i] = ((x - y) * (x - y));
    }
    return neumair_sum(s);
}

/**
 * Computes the cosine distance (not similarity) between {@link a} and {@link b}.
 * @memberof module:metrics
 * @alias cosine
 * @param {Array<Number>} a 
 * @param {Array<Number>} b 
 * @example
 * druid.cosine([1,0],[1,1]) == 0.7853981633974484 == π/4
 * @returns {Number} The cosine distance between {@link a} and {@link b}.
 */
function cosine(a, b) {
    if (a.length !== b.length) return undefined;
    let n = a.length;
    let sum = 0;
    let sum_a = 0;
    let sum_b = 0;
    for (let i = 0; i < n; ++i) {
        sum += (a[i] * b[i]);
        sum_a += (a[i] * a[i]);
        sum_b += (b[i] * b[i]);
    }
    return Math.acos(sum / ((Math.sqrt(sum_a) * Math.sqrt(sum_b))));
}

/**
 * Computes the manhattan distance (l_1) between {@link a} and {@link b}.
 * @memberof module:metrics
 * @alias manhattan
 * @param {Array<Number>} a 
 * @param {Array<Number>} b 
 * @returns {Number} the manhattan distance between {@link a} and {@link b}.  
 */function manhattan(a, b) {
    if (a.length != b.length) return undefined
    let n = a.length;
    let sum = 0;
    for (let i = 0; i < n; ++i) {
        sum += Math.abs(a[i] - b[i]);
    }
    return sum
}

/**
 * Computes the chebyshev distance (l_∞) between {@link a} and {@link b}.
 * @memberof module:metrics
 * @alias chebyshev
 * @param {Array<Number>} a 
 * @param {Array<Number>} b 
 * @returns {Number} the chebyshev distance between {@link a} and {@link b}.  
 */
function chebyshev(a, b) {
    if (a.length != b.length) return undefined
    let n = a.length;
    let res = [];
    for (let i = 0; i < n; ++i) {
        res.push(Math.abs(a[i] - b[i]));
    }
    return Math.max(...res)
}

/**
 * @module metrics
 */

function k_nearest_neighbors(A, k, distance_matrix = null, metric = euclidean) {
    let n = A.length;
    let D = distance_matrix || dmatrix(A, metric);
    for (let i = 0; i < n; ++i) {
        D[i] = D[i].map((d,j) => {
            return {
                i: i, j: j, distance: D[i][j]
            }
        }).sort((a, b) => a.distance - b.distance)
        .slice(1, k + 1);
    }
    return D
}

function dmatrix(A, metric = euclidean) {
    let distance = metric;
    if (distance === undefined) return undefined;
    let n = A.length;
    let D = new Array(n);
    for (let i = 0; i < n; ++i) {
        D[i] = new Array(n);
    }
    for (let i = 0; i < n; ++i) {
        for (let j = i + 1; j < n; ++j) {
            D[i][j] = D[j][i] = distance(A[i], A[j]);
        }
    }
    return D;
}

function linspace(start, end, number = null) {
    if (!number) {
        number = Math.max(Math.round(end - start) + 1, 1);
    }
    if (number < 2) {
        return number === 1 ? [start] : [];
    }
    let result = new Array(number);
    number -= 1;
    for (let i = number; i >= 0; --i) {
        result[i] = (i * end + (number - i) * start) / number;
    }
    return result
}

//import { neumair_sum } from "../numerical/index";

function norm(v, metric = euclidean) {
//export default function(vector, p=2, metric = euclidean) {
    let vector = null;
    if (v instanceof Matrix) {
        let [rows, cols] = v.shape;
        if (rows === 1) vector = v.row(0);
        else if (cols === 1) vector = v.col(0);
        else throw "matrix must be 1d!"
    } else {
        vector = v;
    }
    let n = vector.length;
    let z = new Array(n);
    z.fill(0);
    return metric(vector, z);
    
    
    /*let v;
    if (vector instanceof Matrix) {
        let [ rows, cols ] = v.shape;
        if (rows === 1) {
            v = vector.row(0);
        } else if (cols === 1) {
            v = vector.col(0);
        } else {
            throw "matrix must be 1d"
        }
    } else {
        v = vector;
    }
    return Math.pow(neumair_sum(v.map(e => Math.pow(e, p))), 1 / p)*/
}

/**
 * Computes the QR Decomposition of the Matrix {@link A} using Gram-Schmidt process.
 * @memberof module:linear_algebra
 * @alias qr
 * @param {Matrix} A
 * @returns {{R: Matrix, Q: Matrix}}
 * @see {@link https://en.wikipedia.org/wiki/QR_decomposition#Using_the_Gram%E2%80%93Schmidt_process}
 */
function qr(A) {
    const [rows, cols] = A.shape;
    const Q = new Matrix(rows, cols, "identity");
    const R = new Matrix(cols, cols, 0);

    for (let j = 0; j < cols; ++j) {
        let v = A.col(j);
        for (let i = 0; i < j; ++i) {
            const q = Q.col(i);
            const q_dot_v = neumair_sum(q.map((q_, k) => q_ * v[k]));
            R.set_entry(i,j, q_dot_v);
            v = v.map((v_, k) => v_ - q_dot_v * q[k]);
        }
        const v_norm = norm(v, euclidean);
        for (let k = 0; k < rows; ++k) {
            Q.set_entry(k, j, v[k] / v_norm);
        }
        R.set_entry(j,j, v_norm);
    }
    return {"R": R, "Q": Q};
}

/**
 * Computes the QR Decomposition of the Matrix {@link A} with householder transformations.
 * @memberof module:linear_algebra
 * @alias qr_householder
 * @param {Matrix} A 
 * @returns {{R: Matrix, Q: Matrix}}
 * @see {@link https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections}
 * @see {@link http://mlwiki.org/index.php/Householder_Transformation}
 */
function qr_householder(A) {
    const [rows, cols] = A.shape;
    const Q = new Matrix(rows, rows, "I");
    const R = A.clone();

    for (let j = 0; j < cols; ++j) {
        const x = Matrix.from(R.col(j).slice(j));
        const x_norm = norm(x);
        const x0 = x.entry(0, 0);
        const rho = -Math.sign(x0);
        const u1 = x0 - rho * x_norm;
        const u = x.divide(u1).set_entry(0, 0, 1);
        const beta = -rho * u1 / x_norm;

        const u_outer_u = u.outer(u);
        const R_block = R.get_block(j, 0);
        const new_R = R_block.sub(u_outer_u.dot(R_block).mult(beta));
        const Q_block = Q.get_block(0, j);
        const new_Q = Q_block.sub(Q_block.dot(u_outer_u).mult(beta));
        R.set_block(j, 0, new_R);
        Q.set_block(0, j, new_Q);
        // numerical instability
        for (let k = j + 1; k < rows; ++k) {
            R.set_entry(k, j, 0);
        }
    }
    return {"R": R, "Q": Q};
}

/**
 * Computes the QR Decomposition of the Matrix {@link A} with givens rotation.
 * @memberof module:linear_algebra
 * @alias qr_givens
 * @param {Matrix} A 
 * @returns {{R: Matrix, Q: Matrix}}
 * @see {@link https://en.wikipedia.org/wiki/QR_decomposition#Using_Givens_rotations}
 */
function qr_givens(A) {
    const [rows, cols] = A.shape;
    let Q = new Matrix(rows, rows, "identity");
    let R = A.clone();

    for (let j = 0; j < cols; ++j) {
        //let Gj = new Matrix(rows, rows, "I");
        for (let i = rows - 1; i > j; --i) {
            const [c, s] = givensrotation(R.entry(i - 1, j), R.entry(i, j));
            if (c == 1 && s == 0) continue;
            const Gji = new Matrix(rows, rows, "I");
            Gji.set_entry(i - 1, i - 1, c);
            Gji.set_entry(i - 1, i, -s);
            Gji.set_entry(i, i - 1, s);
            Gji.set_entry(i, i, c);
            R = Gji.T.dot(R);
            Q = Q.dot(Gji);
            //Gj = Gj.dot(Gji)
        }
        /*R = Gj.T.dot(R);
        Q = Q.dot(Gj);*/
        // numerical instability
        for (let k = j + 1; k < rows; ++k) {
            R.set_entry(k, j, 0);
        }
    }
    return {"R": R, "Q": Q};
}

function givensrotation(a, b) {
    if (b == 0) {
        return [1, 0];
    } else {
        if (Math.abs(b) > Math.abs(a)) {
            const r = a / b;
            const s = 1 / Math.sqrt(1 + r ** 2);
            const c = s * r;
            return [c, s];
        } else {
            const r = b / a;
            const c = 1 / Math.sqrt(1 + r ** 2);
            const s = c * r;
            return [c, s];
        }
    }
}

function simultaneous_poweriteration(A, k = 2, max_iterations = 100, seed = 19870307) {
    let randomizer = new Randomizer(seed);
    if (!(A instanceof Matrix)) A = Matrix.from(A);
    let n = A.shape[0];
    let { Q: Q, R: R } = qr(new Matrix(n, k, () => randomizer.random));
    while(max_iterations--) {
        let oldR = R.clone();
        let Z = A.dot(Q);
        let QR = qr(Z);
        [ Q, R ] = [ QR.Q, QR.R ]; 
        if (neumair_sum(R.sub(oldR).diag) / n < 1e-12) {
            max_iterations = 0;
        }        
    }

    let eigenvalues = R.diag;
    let eigenvectors = Q.transpose().to2dArray;//.map((d,i) => d.map(dd => dd * eigenvalues[i]))
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors
    };
}

// crout algorithm
// https://en.wikipedia.org/wiki/Crout_matrix_decomposition
function lu(A) {
    let rows = A.shape[0];
    let L = new Matrix(rows, rows, "zeros");
    let U = new Matrix(rows, rows, "identity");
    let sum;

    for (let j = 0; j < rows; ++j) {
        for (let i = j; i < rows; ++i) {
            sum = 0;
            for (let k = 0; k < j; ++k) {
                sum += L.entry(i, k) * U.entry(k, j);
            }
            /*sum = neumair_sum(linspace(0, j).map((k) => L.entry(i, k) * U.entry(k, j)))
            console.log("lsum1", sum)
            sum = []
            for (let k = 0; k < j; ++k) {
                sum.push(L.entry(i, k) * U.entry(k, j))
            }
            sum = neumair_sum(sum)
            console.log("lsum2", sum)*/
            L.set_entry(i, j, A.entry(i, j) - sum);
        }
        for (let i = j; i < rows; ++i) {
            if (L.entry(j, j) === 0) {
                return undefined;
            }
            sum = 0;
            for (let k = 0; k < j; ++k) {
                sum += L.entry(j, k) * U.entry(k, i);
            }
            /*sum = neumair_sum(linspace(0, j).map((k) => L.entry(j, k) * U.entry(k, i)))
            console.log("usum1", sum)
            sum = []
            for (let k = 0; k < j; ++k) {
                sum.push(L.entry(j, k) * U.entry(k, i))
            }
            sum = neumair_sum("usum2", sum)
            console.log(sum)*/
            U.set_entry(j, i, (A.entry(j, i) - sum) / L.entry(j, j));
        }
    }

    return { L: L, U: U };
}

// doolittle algorithm
/*export default function(M) {
    let [rows, cols] = M.shape;
    let P = new Matrix();
    P.shape = [rows + 1, 1, i => i];
    let A = M.clone();
    let L = new Matrix();
    let U = new Matrix();
    let I = new Matrix();
    I.shape = [rows, rows, (i, j) => i === j ? 1 : 0];

    for (let i = 0; i < rows; ++i) {
        let max_A = 0;
        let i_max = i;
        let abs_A;
        for (let k = i; k < rows; ++k) {
            abs_A = Math.abs(A.entry(k, i))
            if (abs_A > max_A) {
                [ max_A, i_max ] = [ abs_A, k ];
            }

            if (max_A < 1e-12) return undefined;

            if (i_max !== i) {
                let p = P.row(i);
                P.set_row(i, P.row(i_max));
                P.set_row(i_max, p);

                let A_row = A.row(i);
                A.set_row(i, A.row(i_max));
                A.set_row(i_max, A_row);

                P.set_entry(rows + 1, 0, P.entry(rows + 1, 0) + 1)
            }
        }

        for (let j = i + 1; j < rows; ++j) {
            A.set_entry(j, i,  A.entry(j, i) / A.entry(i, i));
            for (let k = i + 1; k < rows; ++k) {
                A.set_entry(j, k, A.entry(j, k) - A.entry(j, i) * A.entry(i, k));
            }
        }
    }

    L.shape = [rows, rows, (i, j) => {
        if ( i > j ) return A.entry(i, j);
        if ( i === j ) return A.entry(i, j) + 1;
        return 0
    }]

    U = A.sub(L)

    return {L: L, U: U, P: P};   
}*/

/**
 * Computes the eigenvector of {@link X} with an accelerated stochastic power iteration algorithm.
 * @memberof module:linear_algebra 
 * @alias svrg
 * @see {@link https://arxiv.org/abs/1707.02670}
 * @param {Matrix} data - the data matrix
 * @param {Matrix} x - Initial Point as 1 times cols Matrix
 * @param {number} beta - momentum parameter
 * @param {number} epoch - number of epochs
 * @param {number} m - epoch length
 * @param {number} s - mini-batch size
 * @param {number} seed - seed for the random number generator
 */
function svrg(data, x, beta, epoch=20, m=10, s=1, seed) {
    let [n, d] = data.shape;
    const randomizer = new Randomizer(seed);
    x = new Matrix(d, 1, () => randomizer.random);
    x = x.divide(norm(x));
    let x0 = x.clone();
    let A = data.T.dot(data).divide(n);
    let x_tilde = x.clone();
    
    for (let t = 0; t < epoch; ++t) {
        const gx = A.dot(x_tilde);
        for (let i = 0; i < m; ++i) {
            const ang = x.T.dot(x_tilde).entry(0, 0);
            const sample = Matrix.from(Randomizer.choice(data, s));
            const sampleT_dot_sample = sample.T.dot(sample);
            const x_tmp = x.clone();
            const XTXx = sampleT_dot_sample
                    .dot(x.divide(s));
            const XTXx_tilde = sampleT_dot_sample
                    .dot(x_tilde.mult(ang / s));
            x = XTXx.sub(XTXx_tilde)
                    .add(gx.mult(ang).sub(x0.mult(beta)));
            x0 = x_tmp;
            const x_norm = norm(x);
            x = x.divide(x_norm);
            x0 = x0.divide(x_norm);        
        }  
        x_tilde = x.clone();
    }
    return x;

}

/**
 * 
 * @param {Matrix} data - the data matrix
 * @param {Matrix} x - Initial Point as 1 times cols Matrix
 * @param {number} beta - momentum parameter
 * @param {number} max_iter - maximum number of iterations
 * @param {number} seed - seed for the random number generator
 */
function poweriteration_m(data, x0, beta, max_iter=20, seed) {
    let randomizer = new Randomizer(seed);
    let [ n, d ] = data.shape;
    let A = data.T.dot(data).divide(n);
    if (x0 === null) x0 = new Matrix(d, 1, () => randomizer.random);
    x0 = x0.divide(norm(x0));
    let x = x0.clone();
    for (let i = 0; i < max_iter; ++i) {
        let x_tmp = x.clone();
        x = A.dot(x).sub(x0.mult(beta));
        x0 = x_tmp;
        let z = norm(x);
        x = x.divide(z);
        x0 = x0.divide(z);
    }
    return x;
}

/**
 * @typedef {Eigenpair} Eigenpair
 * @property {Array} Eigenvalues - Array of Eigenvalues
 * @property {Array[]} Eigenvectors - Array of Eigenvectors 
 */


/**
 * Computes the {@link n} biggest Eigenpair of the Matrix {@link data}.
 * @memberof module:linear_algebra
 * @alias poweriteration_n
 * @param {Matrix} data - the data matrix
 * @param {int} n - Number of Eigenvalues / Eigenvectors
 * @param {Matrix} x - Initial Point as 1 times cols Matrix
 * @param {number} beta - momentum parameter
 * @param {number} max_iter - maximum number of iterations
 * @param {number} seed - seed for the random number generator
 * @returns {Eigenpair} The {@link n} Eigenpairs.
 */
function poweriteration_n(data, n, x0, beta, max_iter=100, seed) {
    const randomizer = new Randomizer(seed);
    const N = data.shape[0];
    //let b = new Matrix(N, n, () => randomizer.random);
    let b = [];

    if (x0 == null) {
        x0 = new Array(n);//new Matrix(N, n, () => randomizer.random)
        
        for (let i = 0; i < n; ++i) {
            x0[i] = new Float64Array(N);
            b[i] = new Float64Array(N);
            for (let j = 0; j < N; ++j) {
                const value = randomizer.random;
                x0[i][j] = value;
                b[i][j] = value;
            }
            let x0_i_norm = norm(x0[i]);
            x0[i] = x0[i].map(x => x / x0_i_norm);
        }
        //x0 = Matrix.from(x0).T;
        //b = Matrix.from(b).T;
    }
    //x0 = x0.divide(norm(x0));
    for (let k = 0; k < n; ++k) {
        let bk = b[k];
        for (let s = 0; s < max_iter; ++s) {
            // Orthogonalize vector
            for (let l = 0; l < k; ++l) {
                const row = b[l];
                const d = neumair_sum((new Float64Array(N)).map((_, i) => bk[i] * row[i]));
                for (let i = 0; i < N; ++i) {
                    bk[i] = bk[i] - (d * row[i]);
                }
            }
            let tmp = data.dot(bk);
            const tmp_norm = norm(tmp);
            x0[k] = tmp.map(t => t / tmp_norm);
            if (neumair_sum((new Float64Array(N)).map((_, i) => tmp[i] * bk[i])) > (1 - 1e-12)) {
                break;
            }
            [bk, tmp] = [tmp, bk];
        }
    }

    return {
        "eigenvalues": b,
        "eigenvectors": x0
    }
}

/**
 * @module linear_algebra
 */

/**
 * @class
 * @alias Matrix
 * @requires module:numerical/neumair_sum
 */
class Matrix{
    /**
     * creates a new Matrix. Entries are stored in a Float64Array. 
     * @constructor
     * @memberof module:matrix
     * @alias Matrix
     * @param {number} rows - The amount of rows of the matrix.
     * @param {number} cols - The amount of columns of the matrix.
     * @param {(function|string|number)} value=0 - Can be a function with row and col as parameters, a number, or "zeros", "identity" or "I", or "center".
     *  - **function**: for each entry the function gets called with the parameters for the actual row and column.
     *  - **string**: allowed are
     *      - "zero", creates a zero matrix.
     *      - "identity" or "I", creates an identity matrix.
     *      - "center", creates an center matrix.
     *  - **number**: create a matrix filled with the given value.
     * @example
     * 
     * let A = new Matrix(10, 10, () => Math.random()); //creates a 10 times 10 random matrix.
     * let B = new Matrix(3, 3, "I"); // creates a 3 times 3 identity matrix.
     * @returns {Matrix} returns a {@link rows} times {@link cols} Matrix filled with {@link value}.
     */
    constructor(rows=null, cols=null, value=null) {
        this._rows = rows;
        this._cols = cols;
        this._data = null;
        if (rows && cols) {
            if (!value) {
                this._data = new Float64Array(rows * cols);
                return this;
            }
            if (typeof(value) === "function") {
                this._data = new Float64Array(rows * cols);
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col < cols; ++col) {
                        this._data[row * cols + col] = value(row, col);
                    }
                }
                return this;
            }
            if (typeof(value) === "string") {
                if (value === "zeros") {
                    return new Matrix(rows, cols, 0); 
                }
                if (value === "identity" || value === "I") {
                    this._data = new Float64Array(rows * cols);
                    for (let row = 0; row < rows; ++row) {
                        this._data[row * cols + row] = 1;
                    }
                    return this;
                }
                if (value === "center" && rows == cols) {
                    this._data = new Float64Array(rows * cols);
                    value = (i, j) => (i === j ? 1 : 0) - (1 / rows);
                    for (let row = 0; row < rows; ++row) {
                        for (let col = 0; col < cols; ++col) {
                            this._data[row * cols + col] = value(row, col);
                        }
                    }
                    return this;
                }
            }
            if (typeof(value) === "number") {
                this._data = new Float64Array(rows * cols);
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col < cols; ++col) {
                        this._data[row * cols + col] = value;
                    }
                }
                return this;
            }
        }
        return this;
        
    }

    /**
     * Creates a Matrix out of {@link A}.
     * @param {(Matrix|Array|Float64Array|number)} A - The matrix, array, or number, which should converted to a Matrix.
     * @param {"row"|"col"|"diag"} [type = "row"] - If {@link A} is a Array or Float64Array, then type defines if it is a row- or a column vector. 
     * @returns {Matrix}
     * 
     * @example
     * let A = Matrix.from([[1, 0], [0, 1]]); //creates a two by two identity matrix.
     * let S = Matrix.from([1, 2, 3], "diag"); // creates a three by three matrix with 1, 2, 3 on its diagonal.
     */
    static from(A, type="row") {
        if (A instanceof Matrix) {
            return A.clone();
        } else if (Array.isArray(A) || A instanceof Float64Array) {
            let m = A.length;
            if (m === 0) throw "Array is empty";
            // 1d
            if (!Array.isArray(A[0]) && !(A[0] instanceof Float64Array)) {
                if (type === "row") {  
                    return new Matrix(1, m, (_, j) => A[j]);
                } else if (type === "col") {
                    return new Matrix(m, 1, (i) => A[i]);
                } else if (type === "diag") {
                    return new Matrix(m, m, (i, j) => (i == j) ? A[i] : 0);
                } else {
                    throw "1d array has NaN entries"
                }
            // 2d
            } else if (Array.isArray(A[0]) || A[0] instanceof Float64Array) {
                let n = A[0].length;
                for (let row = 0; row < m; ++row) {
                    if (A[row].length !== n) throw "various array lengths";
                }
                return new Matrix(m, n, (i, j) => A[i][j])
            }
        } else if (typeof(A) === "number") {
            return new Matrix(1, 1, A);
        } else {
            throw "error"
        }
    }

    /**
     * Returns the {@link row}th row from the Matrix.
     * @param {int} row 
     * @returns {Array}
     */
    row(row) {
        let result_row = new Array(this._cols);
        for (let col = 0; col < this._cols; ++col) {
            result_row[col] = this._data[row * this._cols + col];
        }
        return result_row;
    }

    /**
     * Sets the entries of {@link row}th row from the Matrix to the entries from {@link values}.
     * @param {int} row 
     * @param {Array} values 
     * @returns {Matrix}
     */
    set_row(row, values) {
        let cols = this._cols;
        if (Array.isArray(values) && values.length === cols) {
            let offset = row * cols;
            for (let col = 0; col < cols; ++col) {
                this._data[offset + col] = values[col];
            }
        } else if (values instanceof Matrix && values.shape[1] === cols && values.shape[0] === 1) {
            let offset = row * cols;
            for (let col = 0; col < cols; ++col) {
                this._data[offset + col] = values._data[col];
            }
        }
        return this;
    }

    /**
     * Returns the {@link col}th column from the Matrix.
     * @param {int} col 
     * @returns {Array}
     */
    col(col) {
        let result_col = new Array(this._rows);
        for (let row = 0; row < this._rows; ++row) {
            result_col[row] = this._data[row * this._cols + col];
        }
        return result_col;
    }

    /**
     * Returns the {@link col}th entry from the {@link row}th row of the Matrix.
     * @param {int} row 
     * @param {int} col 
     * @returns {float64}
     */
    entry(row, col) {
        return this._data[row * this._cols + col];
    }

    /**
     * Sets the {@link col}th entry from the {@link row}th row of the Matrix to the given {@link value}.
     * @param {int} row 
     * @param {int} col 
     * @param {float64} value
     * @returns {Matrix}
     */
    set_entry(row, col, value) {
        this._data[row * this._cols + col] = value;
        return this;
    }

    /**
     * Returns a new transposed Matrix.
     * @returns {Matrix}
     */
    transpose() {
        let B = new Matrix(this._cols, this._rows, (row, col) => this.entry(col, row));
        return B;
    }

    /**
     * Returns a new transposed Matrix. Short-form of {@function transpose}.
     * @returns {Matrix}
     */
    get T() {
        return this.transpose();
    }

    /**
     * Returns the inverse of the Matrix.
     * @returns {Matrix}
     */
    inverse() {
        const rows = this._rows;
        const cols = this._cols;
        let B = new Matrix(rows, 2 * cols, (i,j) => {
            if (j >= cols) {
                return (i === (j - cols)) ? 1 : 0;
            } else {
                return this.entry(i, j);
            }
        });
        let h = 0; 
        let k = 0;
        while (h < rows && k < cols) {
            var i_max = 0;
            let max_val = -Infinity;
            for (let i = h; i < rows; ++i) {
                let val = Math.abs(B.entry(i,k));
                if (max_val < val) {
                    i_max = i;
                    max_val = val;
                }
            }
            if (B.entry(i_max, k) == 0) {
                k++;
            } else {
                // swap rows
                for (let j = 0; j < 2 * cols; ++j) {
                    let h_val = B.entry(h, j);
                    let i_val = B.entry(i_max, j);
                    B.set_entry(h, j, h_val);
                    B.set_entry(i_max, j, i_val);
                }
                for (let i = h + 1; i < rows; ++i) {
                    let f = B.entry(i, k) / B.entry(h, k);
                    B.set_entry(i, k, 0);
                    for (let j = k + 1; j < 2 * cols; ++j) {
                        B.set_entry(i, j, B.entry(i, j) - B.entry(h, j) * f);
                    }
                }
                h++;
                k++;
            }
        }

        for (let row = 0; row < rows; ++row) {
            let f = B.entry(row, row);
            for (let col = row; col < 2 * cols; ++col) {
                B.set_entry(row, col, B.entry(row, col) / f);
            }
        }
        
        for (let row = rows - 1; row >= 0; --row) {
            let B_row_row = B.entry(row, row);
            for (let i = 0; i < row; i++) {
                let B_i_row = B.entry(i, row);
                let f = B_i_row / B_row_row;
                for (let j = i; j < 2 * cols; ++j) {
                    let B_i_j = B.entry(i,j);
                    let B_row_j = B.entry(row, j);
                    B_i_j = B_i_j - B_row_j * f;
                    B.set_entry(i, j, B_i_j);
                }
            }
        }

        return new Matrix(rows, cols, (i,j) => B.entry(i, j + cols));
    }

    /**
     * Returns the dot product. If {@link B} is an Array or Float64Array then an Array gets returned. If {@link B} is a Matrix then a Matrix gets returned.
     * @param {(Matrix|Array|Float64Array)} B the right side
     * @returns {(Matrix|Array)}
     */
    dot(B) {
        if (B instanceof Matrix) {
            let A = this;
            if (A.shape[1] !== B.shape[0]) {
                throw `A.dot(B): A is a ${A.shape.join(" x ")}-Matrix, B is a ${B.shape.join(" x ")}-Matrix: 
                A has ${A.shape[1]} cols and B ${B.shape[0]} rows. 
                Must be equal!`;
            }
            let I = A.shape[1];
            let C = new Matrix(A.shape[0], B.shape[1], (row, col) => {
                let A_i = A.row(row);
                let B_i = B.col(col);
                for (let i = 0; i < I; ++i) {
                    A_i[i] *= B_i[i];
                }
                return neumair_sum(A_i);
            });
            return C;
        } else if (Array.isArray(B) || (B instanceof Float64Array)) {
            let rows = this._rows;
            if (B.length !== rows)  {
                throw `A.dot(B): A has ${rows} cols and B has ${B.length} rows. Must be equal!`
            }
            let C = new Array(rows);
            for (let row = 0; row < rows; ++row) {
                C[row] = neumair_sum(this.row(row).map(e => e * B[row]));
            }
            return C;
        } else {
            throw `B must be Matrix or Array`;
        }
    }

    /**
     * Computes the outer product from {@link this} and {@link B}.
     * @param {Matrix} B 
     * @returns {Matrix}
     */
    outer(B) {
        let A = this;
        let l = A._data.length;
        let r = B._data.length;
        if (l != r) return undefined;
        let C = new Matrix();
        C.shape = [l, l, (i, j) => {
            if (i <= j) {
                return A._data[i] * B._data[j];
            } else {
                return C.entry(j, i);
            }
        }];
        return C;
    }

    /**
     * 
     * @param {Matrix} A - Matrix to bidiagonlize.
     * @param {Number} m - number of bidiagonalization steps.
     * 
     */
    static lanczos_bidiagonal(A, m, p_0 = null) {
        const append_col = (A, v) => {
            const [rows, cols] = A.shape;
            return new Matrix(rows, cols + 1, (i, j) => {
                if (j < cols) {
                    return A.entry(i, j);
                } else {
                    return v.entry(i, 0);
                }
            })
        };
        const [l, n] = A.shape;
        const AT = A.T;

        if (!p_0) {
            p_0 = new Matrix(n, 1, () => Math.random());
            p_0 = p_0.divide(norm(p_0));
        }

        const B = new Matrix(m, m, () => 0);
        const alpha = [];

        // 1
        let P = p_0;
        let q_0 = A.dot(p_0);
        // 2
        alpha.push(norm(q_0));
        q_0 = q_0.divide(alpha[0]);
        let Q = q_0;
        B.set_entry(0, 0, alpha[0]);
        let r_j = null;
        // 3
        for (let j = 0; j < m ; ++j) {
            const alpha_j = alpha[j];
            const q_j = Matrix.from(Q.col(j), "col");
            const p_j = Matrix.from(P.col(j), "col");
            // 4
            r_j = AT.dot(q_j).sub(p_j.mult(alpha_j));
            // 5 Reorthogonalization
            r_j = r_j.sub(P.dot(P.T.dot(r_j)));
            // 6
            if (j < m - 1) {
                // 7
                const beta_j = norm(r_j);
                const p_j1 = r_j.divide(beta_j);
                P = append_col(P, p_j1);
                // 8
                let q_j1 = A.dot(p_j1).sub(q_j.mult(beta_j));
                // 9 Reorthogonalization
                q_j1 = q_j1.sub(Q.dot(Q.T.dot(q_j1)));
                // 10
                const alpha_j1 = norm(q_j1);
                alpha.push(alpha_j1);
                q_j1 = q_j1.divide(alpha_j1);
                
                Q = append_col(Q, q_j1);
                B.set_entry(j, j + 1, beta_j);
                B.set_entry(j + 1, j + 1, alpha_j1);
            // 11 endif
            }
        // 12 endfor
        }
        return [P, Q, B, r_j];
    }

    /**
     * Augmented implicitly restarted Lanczos bidiagonalization method.
     * @param {Matrix} M - a l x n matrix. If l < n it computes SVD of M transposed.
     * @param {Number} k - number of desired singular triplets.
     * @param {Number} m - number of bidiagonalization steps.
     * @param {Number} [tol = 1e-3] - tolerance for acception computed approximate singular triplet
     * @param {Number} [eps = 1e-8] - machine epsilon
     * @param {Boolean} [harmonic = true] - suggests type of augmentation.
     * @returns {Object} Truncated Singular Value Decomposition. Returns U {@link k} singular left vectors, Sigma {@link k} singular values, V {@link k} singular right vectors.
     * @see {@link http://www.math.kent.edu/~reichel/publications/auglbd.pdf}
     * @see {@link https://en.wikipedia.org/wiki/Singular_value_decomposition}
     */
    static augmented_lanczos_bidiagonalization(M, k, m, tol = 1e-3, eps = 1e-8, harmonic = true) {
        // helper function
        const eig_svd = (B) => {
            const n = B.shape[1];
            const BT = B.T;
            const BTB = BT.dot(B);
            const BBT = B.dot(BT);
            const {eigenvectors: V, eigenvalues: S} = simultaneous_poweriteration(BTB, n);
            const {eigenvectors: U} = simultaneous_poweriteration(BBT, n);
            return [
                Matrix.from(U), 
                S.map(s => Math.sqrt(s)), 
                Matrix.from(V)
            ];
        };
    
        // helper function
        const check_convergence = (U, B) => {
            const A_norm = B.diag.sort()[m - 1]; // matrix norm... largest singular value
            const max = tol * A_norm;
            const beta_m = B.entry(m - 1, m);
            let converged = true;
            for (let j = 0; j < m; ++j) {
                converged = converged && (beta_m * Math.abs(U.entry(j, j)) <= max);
            }
            return converged;
        };

        // helper function
        const append_col = (A, v) => {
            const [rows, cols] = A.shape;
            return new Matrix(rows, cols + 1, (i, j) => (j < cols) ? A.entry(i, j) : v.entry(i, 0));
        };
        
        const [rows, cols] = M.shape;
        const A = rows > cols ? M : M.T;
        const AT = rows > cols ? M.T : M;
        const [l, n] = A.shape;

        // init vector;
        let p_0 = new Matrix(n, 1, () => Math.random());
        p_0 = p_0.divide(norm(p_0));

        // 1. compute the partial lanczos bidiagonalization using lanczos_bidiagonalization;
        let [P, Q, B, r] = Matrix.lanczos_bidiagonal(A, m, p_0);
        // (1.3) (1.4)
        // console.log(A.dot(P).sub(Q.dot(B)))
        // A.dot(P) = Q.dot(B) .... almost :-)
        //const emT = new Matrix(1, m, () => 0).set_entry(0, m - 1, 1)
        //console.log(AT.dot(Q), P.dot(B.T).sub(r.dot(emT)), AT.dot(Q).sub(P.dot(B.T).sub(r.dot(emT))));
        // A.T.dot(Q) = P.dot(B.T)+rm.dot(emT) ... nope =/
        let U, S, V;
        let max_iter = 10;
        while(--max_iter) {
            console.log(max_iter);
            // 2. compute the singular value decomposition of B
            [U, S, V] = eig_svd(B);
            // K condition number: biggest singular value / smallest singular value
            const K = (S) => {
                console.log(S);
                return S[0] / S[S.length - 1];
            };
            // 3. check convergence: if all k desired singular triplets satisfy (2.13), then exit
            console.log(check_convergence(U, B));
            if (check_convergence(U, B)) {
                console.log("converged");
                return [U, S, V]
            } else {
                // 4. compute the augmenting vectors:
                console.log(K(S), ">", Math.pow(eps, -.5), K(S) > Math.pow(eps, -.5));
                const beta_m = B.entry(k - 1, k);
                if (!harmonic || K(S) > Math.pow(eps, -.5)) {
                    // 4a. determine P, Q, B, and r by (3.2), (3.5), (3.6), (3.10) respectively.
                    console.log("4a");
                    const P = new Matrix(V.shape[0], k + 1, (i, j) => (j < k + 1) ? V.entry(i, j) : P.entry(i, 0));
                    const r_norm = norm(r);
                    Q = new Matrix(U.shape[0], k + 1, (i, j) => (j < k + 1) ? U.entry(i, j) : (r.entry(i, 0) / r_norm));
                    //const alpha_k1 = B.entry(k + 1, k + 1);
                    B = new Matrix(k + 1, k + 1, (i, j) => {
                        if (j < k + 1) {
                            return (i == j) ? S[i] : 0;
                        } else {
                            return (i < k + 1) ? (beta_m * U.entry(i, i)): B.entry(i, j);
                        }
                    });
                    const f = 
                    console.log(P, Q, B);
                } else { // harmonic && K(S) <= Math.pow(eps, -.5)
                    // 4b. compute the partial singular value decompostion (3.24 of B_m,m+1)
                    // and the QR-factorization (3.28);
                    // determine P, Q, B, r by (3.29), (3.32), (3.33), and (3.39) respectively
                    console.log("4b");
                    // compute partial singular value decomposition (3.24) of B_mm1
                    const B_mm1 = new Matrix(m - 1, m, (i, j) => B.entry(i, j));
                    let [U_prime, S_prime, V_prime] = eig_svd(B_mm1);
                    S_prime = Matrix.from(S_prime.slice(0, k), "diag"); //new Matrix(k, k, (i, j) => (i == j) ? S[i] : 0);
                    const B_inv = B_mm1.inverse();
                    const B_inv_U_prime_S_prime = B_inv.dot(U_prime).dot(S_prime);
                    // compute QR-factorization (3.28)
                    const {"Q": Q_prime, "R": R_prime} = qr(new Matrix(k + 1, k + 1, (i, j) => {
                            if (i < k) {
                                return (j < k) ? B_inv_U_prime_S_prime.entry(i, j) : -beta_m * B_inv.entry(i, j);
                            } else {
                                return (j < k) ? 0 : 1;
                            }
                        })
                    );
                    // determine P, Q, B, and r
                    // P
                    const P_hat_k1 = P.dot(Q_prime);
                    // Q
                    const Q_hat_k = Q.dot(U_prime);
                    const q_m = Matrix.from(Q.col(m - 2), "col");
                    const p_m1 = Matrix.from(P.col(m - 1), "col");
                    const q_m_beta_m = q_m.mult(-beta_m);
                    const A_p_m1 = A.dot(p_m1);
                    const q_m_beta_m_add_A_p_m1 = q_m_beta_m.add(A_p_m1);
                    const c_k = Q_hat_k.T.dot(q_m_beta_m_add_A_p_m1);
                    const alpha_k1 = B.entry(k, k);
                    const q_k1 = q_m_beta_m_add_A_p_m1.sub(Q_hat_k.dot(c_k)).divide(alpha_k1);
                    const Q_hat_k1 = append_col(Q_hat_k, q_k1);
                    // B
                    const B_hat_k1 = new Matrix(k + 1, k + 1, (i, j) => {
                        if (i < k) {
                            return (j < k) ? (i == j ? S_prime.entry(i, i) : 0) : c_k.entry(i, 0);
                        } else {
                            return (j < k) ? 0 : alpha_k1;
                        }
                    }).dot(R_prime.inverse());
                    // r
                    const r_breve = AT.dot(q_k1).sub(p_m1.mult(alpha_k1));

                    // 5. Append...
                    P.set_block(0, 0, P_hat_k1);
                    Q.set_block(0, 0, Q_hat_k1);
                    B.set_block(0, 0, B_hat_k1);
                    r = r_breve;
                }
                // 5.
            }
        }

        return eig_svd(B)//{ "U": P, "Sigma": B, "V": Q };
    }

    /**
     * Appends matrix {@link B} to the matrix.
     * @param {Matrix} B - matrix to append.
     * @param {"horizontal"|"vertical"|"diag"} [type = "horizontal"] - type of concatenation.
     * @returns {Matrix}
     * @example
     * 
     * let A = Matrix.from([[1, 1], [1, 1]]); // 2 by 2 matrix filled with ones.
     * let B = Matrix.from([[2, 2], [2, 2]]); // 2 by 2 matrix filled with twos.
     * 
     * A.concat(B, "horizontal"); // 2 by 4 matrix. [[1, 1, 2, 2], [1, 1, 2, 2]]
     * A.concat(B, "vertical"); // 4 by 2 matrix. [[1, 1], [1, 1], [2, 2], [2, 2]]
     * A.concat(B, "diag"); // 4 by 4 matrix. [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]
     */
    concat(B, type="horizontal") {
        const A = this;
        const [rows_A, cols_A] = A.shape;
        const [rows_B, cols_B] = B.shape;
        if (type == "horizontal") {
            if (rows_A != rows_B) throw `A.concat(B, "horizontal"): A and B need same number of rows, A has ${rows_A} rows, B has ${rows_B} rows.`;
            const X = new Matrix(rows_A, cols_A + cols_B, "zeros");
            X.set_block(0, 0, A);
            X.set_block(0, cols_A, B);
            return X;
        } else if (type == "vertical") {
            if (cols_A != cols_B) throw `A.concat(B, "vertical"): A and B need same number of columns, A has ${cols_A} columns, B has ${cols_B} columns.`;
            const X = new Matrix(rows_A + rows_B, cols_A, "zeros");
            X.set_block(0, 0, A);
            X.set_block(rows_A, 0, B);
            return X;
        } else if (type == "diag") {
            const X = new Matrix(rows_A + rows_B, cols_A + cols_B, "zeros");
            X.set_block(0, 0, A);
            X.set_block(rows_A, cols_A, B);
            return X;
        } else {
            throw `type must be "horizontal" or "vertical", but type is ${type}!`;
        }
    }

    static irlbablk(M, k, adjust = 3, block_size = 2, m = 10, tol = 1e-3, eps = 1e-8, harmonic = true) {
        
        const [rows, cols] = M.shape;
        const A = rows > cols ? M : M.T;
        const AT = rows > cols ? M.T : M;
        const [p, n] = A.shape;

        if (block_size * m >= n) {
            m = Math.floor((n - block_size - .1) / block_size);
            if (block_size * m - k - block_size < 0) {
                adjust = 0; 
                k = block_size * m - block_size;
            }
        }
        if (block_size * m - k - block_size < 0) {
            m = Math.ceil((k + block_size) / block_size + block_size + .1);
        }
        if (block_size * m >= n) {
            m = Math.floor((n - block_size - .1) / block_size);
            adjust = 0;
            k = block_size * m - block_size;    
        }

        // block lanczos bidiagonalization decomposition
        const ablanzbd = (A, P_r) => {
            const m_b = m * block_size;
            const B = new Matrix(m_b, m_b, "zeros");
            const P = new Matrix(n, m_b, "zeros");
            const Q = new Matrix(p, m_b, "zeros");
            let F_r = new Matrix(n, block_size, "zeros");

            P.set_block(0, 0, P_r);
            let W_r = A.dot(P_r);
            const {"Q": Q_1, "R": S_1} = qr(W_r);
            console.log(S_1, Q_1);
            B.set_entry(0, 0, S_1);
            Q.set_entry(0, 0, Q_1);
            for (let j = 0; j < m; ++j) {
                console.log(j, m);
                const j_b = j * block_size;
                const j1_b = j_b + block_size;
                const P_j = P.get_block(0, j_b, null, j1_b);
                const P_jr = P.get_block(0, 0, null, j1_b);
                const Q_j = Q.get_block(0, j_b, null, j1_b);
                const Q_jr = Q.get_block(0, 0, null, j1_b);
                const S_j = B.get_block(j_b, j_b, j1_b, j1_b);
                console.log(P_j, P_jr, Q_j, Q_jr);
                F_r = AT.dot(Q_j).sub(P_j.dot(S_j.T));
                // Reorthogonalization
                console.log(P_jr.T, F_r);
                F_r = F_r.sub(P_jr.dot(P_jr.T.dot(F_r)));
                if (j < m) {
                    const {"Q": P_j1, "R": R_j1} = qr(F_r);
                    const L_j1 = R_j1.T;
                    P.set_entry(0, j1_b, P_j1);
                    B.set_entry(j_b, j1_b, L_j1);
                    W_r = A.dot(P_j1).sub(Q_j.dot(L_j1));
                    //Reorthogonalization
                    W_r = W_r.sub(Q_jr.dot(Q_jr.T.dot(W_r)));
                    const {"Q": Q_j1, "R": S_j1} = qr(W_r);
                    Q.set_block(0, j1_b, Q_j1);
                    B.set_block(j1_b, j1_b, S_j1);
                }
            }
            return [P, Q, B, F_r];
        };

        // preallocate memeory for W and F
        const W = new Matrix(p, block_size * m, "zeros");
        const F = new Matrix(n, block_size, "zeros");

        // starting matrix
        const V = new Matrix(n, block_size * m, (i, j) => (j < block_size) ? Math.random() : 0);

        // set tolerance to machine precision /** @todo machine precision? */
        if (tol < eps) tol = eps;

        // B            B           block diagonal matrix
        // B_sz         B_shape     size of block diagonal matrix
        // conv         conv        boolean determines convergence
        // eps23        -           used for Smax
        // I            indices     used for indexing
        // iter         -           iteration count
        // J            jndices     used for indexing
        // mprod        -           number of matrix vector products with A and AT
        // R_F          R_F         R fro QR factorization of the resdual matrix F
        // sqrteps      -           square root of machine tolerance used in convergence ttesting
        // Smax         S_max       max singular value of B
        // Smin         S_min       min singular value of B
        // SVTol        -           tolerance to determine when a singular vector has converged
        // S_B          S_B         singular values of B
        // U_B          U_B         left sing. vectors of B
        // V_B          V_B         right sing. vectors of B
        // V_B_last     V_B_last    last row of modified V_B
        // S_B2         S_B2        sing. values of B.concat(E_m*R_F.T)
        // U_B2         U_B2        left sing. vectors of ^^^
        // V_B2         V_B2        right sing. vectors of ^^
        
        console.log(ablanzbd(A, V));
    }





    /**
     * Transforms A to bidiagonal form. (With Householder transformations)
     * @param {Matrix} A 
     */
    static bidiagonal(A, L, R) {
        const get_householder_vector = (x) => {
            const x_norm = norm(x);
            const x0 = x.entry(0, 0);
            const rho = -Math.sign(x0);
            const u1 = x0 - rho * x_norm;
            const u = x.divide(u1).set_entry(0, 0, 1);
            const beta = -rho * u1 / x_norm;
            return u.outer(u).mult(beta);
        };

        const block_dot = (M, r, c, uuT, T) => {
            const M_block = M.get_block(r, c);
            const new_M = T ? M_block.sub(M_block.dot(uuT)) : M_block.sub(uuT.dot(M_block));
            //console.log(T, M_block.shape, uuT.shape, T ? M_block.dot(uuT) : uuT.dot(M_block))
            M.set_block(r, c, new_M);
        };

        A = A.clone();
        const [ rows, cols ] = A.shape;
        for (let k = 0; k < cols; ++k) {
            const x = Matrix.from(A.col(k).slice(k));
            const uuT = get_householder_vector(x);
            block_dot(A, k, k, uuT, false);
            if (L) block_dot(L, k, k, uuT, false);
            if (R) block_dot(R, k, k, uuT, true);
            // repair numerical unstability?            
            for (let r = k + 1; r < rows; ++r) {
                A.set_entry(r, k, 0);
            }
            if (k <= cols - 2) {
                const y = Matrix.from(A.row(k).slice(k + 1));
                const vvT = get_householder_vector(y);
                block_dot(A, k, k + 1, vvT, true);
                if (L) block_dot(L, k, k + 1, vvT, true);
                if (R) block_dot(R, k + 1, k, vvT, false);
                // repair numerical unstability?
                for (let c = k + 2; c < cols; ++c) {
                    A.set_entry(k, c, 0);
                }
            }
        }
        return A;

        /*A = A.clone()
        const [ rows, cols ] = A.shape;
        
        const get_householder_matrix = (M, k, T) => {
            const k1 = T ? k + 1 : k;
            const x = Matrix.from(T ? M.row(k) : M.col(k));
            const x_norm = norm(x);
            const x0 = x.entry(k1, 0);
            const rho = -Math.sign(x0);
            const u1 = x0 - rho * x_norm;
            const u = x.divide(u1).set_entry(k1, 0, 1);
            const beta = -rho * u1 / x_norm;
            const beta_uuT = u.outer(u).mult(beta)
            console.log(cols, beta_uuT)
            return new Matrix(cols, cols, "I").sub(beta_uuT);
        }

        for (let k = 0; k < cols; ++k) {
            const H_c = get_householder_matrix(A, k, false);
            A = H_c.dot(A); //block_dot(A, k, k, uuT, false);
            if (L) L = H_c.dot(L); //block_dot(L, k, k, uuT, false);
            if (R) R = R.dot(H_c);//block_dot(R, k, k, uuT, true);
            // repair numerical unstability?            
            for (let r = k + 1; r < rows; ++r) {
                A.set_entry(r, k, 0);
            }
            if (k <= cols - 2) {
                const H_r = get_householder_matrix(A, k, true);
                A = A.dot(H_r);
                if (L) L = L.dot(H_r);// block_dot(L, k, k + 1, vvT, true);
                if (R) R = H_r.dot(R); //block_dot(R, k + 1, k, vvT, false);
                // repair numerical unstability?
                for (let c = k + 2; c < cols; ++c) {
                    A.set_entry(k, c, 0);
                }
            }
        }
        return A;*/
    }

    /**
     * Writes the entries of B in A at an offset position given by {@link offset_row} and {@link offset_col}.
     * @param {int} offset_row 
     * @param {int} offset_col 
     * @param {Matrix} B 
     * @returns {Matrix}
     */
    set_block(offset_row, offset_col, B) {
        let [ rows, cols ] = B.shape;
        for (let row = 0; row < rows; ++row) {
            if (row > this._rows) continue;
            for (let col = 0; col < cols; ++col) {
                if (col > this._cols) continue;
                this.set_entry(row + offset_row, col + offset_col, B.entry(row, col));
            }
        }
        return this;
    }

    /**
     * Extracts the entries from the {@link start_row}th row to the {@link end_row}th row, the {@link start_col}th column to the {@link end_col}th column of the matrix.
     * If {@link end_row} or {@link end_col} is empty, the respective value is set to {@link this.rows} or {@link this.cols}.
     * @param {Number} start_row 
     * @param {Number} start_col
     * @param {Number} [end_row = null]
     * @param {Number} [end_col = null] 
     * @returns {Matrix} Returns a end_row - start_row times end_col - start_col matrix, with respective entries from the matrix.
     * @example
     * 
     * let A = Matrix.from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]); // a 3 by 3 matrix.
     * 
     * A.get_block(1, 1).to2dArray; // [[5, 6], [8, 9]]
     * A.get_block(0, 0, 1, 1).to2dArray; // [[1]]
     * A.get_block(1, 1, 2, 2).to2dArray; // [[5]]
     * A.get_block(0, 0, 2, 2).to2dArray; // [[1, 2], [4, 5]]
     */
    get_block(start_row, start_col, end_row = null, end_col = null) {
        const [ rows, cols ] = this.shape;
        /*if (!end_row)) {
            end_row = rows;
        }
            end_col = cols;
        }*/
        end_row = end_row || rows;
        end_col = end_col || cols;
        if (end_row <= start_row || end_col <= start_col) {
            throw `
                end_row must be greater than start_row, and 
                end_col must be greater than start_col, but
                end_row = ${end_row}, start_row = ${start_row}, end_col = ${end_col}, and start_col = ${start_col}!`;
        }
        const X = new Matrix(end_row - start_row, end_col - start_col, "zeros");
        for (let row = start_row, new_row = 0; row < end_row; ++row, ++new_row) {
            for (let col = start_col, new_col = 0; col < end_col; ++col, ++new_col) {
                X.set_entry(new_row, new_col, this.entry(row, col));
            }
        }
        return X;
        //return new Matrix(end_row - start_row, end_col - start_col, (i, j) => this.entry(i + start_row, j + start_col));
    }

    /**
     * Applies a function to each entry of the matrix.
     * @param {function} f function takes 2 parameters, the value of the actual entry and a value given by the function {@link v}. The result of {@link f} gets writen to the Matrix.
     * @param {function} v function takes 2 parameters for row and col, and returns a value witch should be applied to the colth entry of the rowth row of the matrix.
     */
    _apply_array(f, v) {
        const data = this._data;
        const [ rows, cols ] = this.shape;
        for (let row = 0; row < rows; ++row) {
            const o = row * cols;
            for (let col = 0; col < cols; ++col) {
                const i = o + col;
                const d = data[i];
                data[i] = f(d, v(row, col));
            }
        }
        return this; 
    }

    _apply_rowwise_array(values, f) {
        return this._apply_array(f, (i, j) => values[j]);
    }

    _apply_colwise_array(values, f) {
        const data = this._data;
        const [ rows, cols ] = this.shape;
        for (let row = 0; row < rows; ++row) {
            const o = row * cols;
            for (let col = 0; col < cols; ++col) {
                const i = o + col;
                const d = data[i];
                data[i] = f(d, values[row]);
            }
        }
        return this; 
    }

    /*_apply_pointwise_number(value, f) {

    }*/

    _apply(value, f) {
        let data = this._data;
        if (value instanceof Matrix) {
            let [ value_rows, value_cols ] = value.shape;
            let [ rows, cols ] = this.shape;
            if (value_rows === 1) {
                if (cols !== value_cols) {
                    throw `cols !== value_cols`;
                }
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col < cols; ++col) {
                        data[row * cols + col] = f(data[row * cols + col], value.entry(0, col));
                    }
                }
            } else if (value_cols === 1) {
                if (rows !== value_rows) {
                    throw `rows !== value_rows`;
                }
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col < cols; ++col) {
                        data[row * cols + col] = f(data[row * cols + col], value.entry(row, 0));
                    }
                }
            } else if (rows == value_rows && cols == value_cols) {
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col < cols; ++col) {
                        data[row * cols + col] = f(data[row * cols + col], value.entry(row, col));
                    }
                }
            } else {
                throw `error`;
            }
        } else if (Array.isArray(value)) {
            let rows = this._rows;
            let cols = this._cols;
            if (value.length === rows) {
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col < cols; ++col) {
                        data[row * cols + col] = f(data[row * cols + col], value[row]);
                    }
                }
            } else if (value.length === cols) {
                for (let row = 0; row < rows; ++row) {
                    for (let col = 0; col < cols; ++col) {
                        data[row * cols + col] = f(data[row * cols + col], value[col]);
                    }
                }
            } else {
                throw `error`;
            }
        } else {
            for (let i = 0, n = this._rows * this._cols; i < n; ++i) {
                data[i] = f(data[i], value);
            }
        }
        return this;
    }

    /**
     * Clones the Matrix.
     * @returns {Matrix}
     */
    clone() {
        let B = new Matrix();
        B._rows = this._rows;
        B._cols = this._cols;
        B._data = this._data.slice(0);
        return B;
    }

    mult(value) {
        return this.clone()._apply(value, (a,b) => a * b);
    }

    divide(value) {
        return this.clone()._apply(value, (a,b) => a / b);
    }

    add(value) {
        return this.clone()._apply(value, (a,b) => a + b);
    }

    sub(value) {
        return this.clone()._apply(value, (a,b) => a - b);
    }

    /**
     * Returns the number of rows and columns of the Matrix.
     * @returns {Array} An Array in the form [rows, columns].
     */
    get shape() {
        return [this._rows, this._cols];
    }

    /**
     * Returns the matrix in the given shape with the given function which returns values for the entries of the matrix.
     * @param {Array} parameter - takes an Array in the form [rows, cols, value], where rows and cols are the number of rows and columns of the matrix, and value is a function which takes two parameters (row and col) which has to return a value for the colth entry of the rowth row.
     * @returns {Matrix}
     */
    set shape([ rows, cols, value = () => 0 ]) {
        this._rows = rows;
        this._cols = cols;
        this._data = new Float64Array(rows * cols);
        for (let row = 0; row < rows; ++row) {
            for (let col = 0; col < cols; ++col) {
                this._data[row * cols + col] = value(row, col);
            }
        }
        return this;
    }

    /**
     * Returns the Matrix as a two-dimensional Array.
     * @returns {Array}
     */
    get to2dArray() {
        const rows = this._rows;
        const cols = this._cols;
        let result = new Array(rows);
        for (let row = 0; row < rows; ++row) {
            let result_col = new Array(cols);
            for (let col = 0; col < cols; ++col) {
                result_col[col] = this.entry(row, col);
            }
            result[row] = result_col;
        }
        return result;
    }

    /**
     * Returns the diagonal of the Matrix.
     * @returns {Array}
     */
    get diag() {
        const rows = this._rows;
        const cols = this._cols;
        const min_row_col = Math.min(rows, cols);
        let result = new Array(min_row_col);
        for (let i = 0; i < min_row_col; ++i) {
            result[i] = this.entry(i,i);
        }
        return result;
    }

    /**
     * Returns the mean of all entries of the Matrix.
     * @returns {float64}
     */
    get mean() {
        const data = this._data;
        const n = this._rows * this._cols;
        let sum = 0;
        for (let i = 0; i < n; ++i) {
            sum += data[i];
        }
        return sum / n;
    }

    /**
     * Returns the mean of each row of the matrix.
     * @returns {Array}
     */
    get meanRows() {
        const data = this._data;
        const rows = this._rows;
        const cols = this._cols;
        let result = [];
        for (let row = 0; row < rows; ++row) {
            result[row] = 0;
            for (let col = 0; col < cols; ++col) {
                result[row] += data[row * cols + col];
            }
            result[row] /= cols;
        }
        return result;
    }

    /** Returns the mean of each column of the matrix.
     * @returns {Array}
     */
    get meanCols() {
        const data = this._data;
        const rows = this._rows;
        const cols = this._cols;
        let result = [];
        for (let col = 0; col < cols; ++col) {
            result[col] = 0;
            for (let row = 0; row < rows; ++row) {
                result[col] += data[row * cols + col];
            }
            result[col] /= rows;
        }
        return result;
    }

    /**
     * Solves the equation {@link A}x = {@link b}. Returns the result x.
     * @param {Matrix} A - Matrix
     * @param {Matrix} b - Matrix
     * @returns {Matrix}
     */
    static solve(A, b) {
        let rows = A.shape[0];
        let { L: L, U: U } = Matrix.LU(A);//lu(A);
        let x = b.clone();
        
        // forward
        for (let row = 0; row < rows; ++row) {
            for (let col = 0; col < row - 1; ++col) {
                x.set_entry(0, row, x.entry(0, row) - L.entry(row, col) * x.entry(1, col));
            }
            x.set_entry(0, row, x.entry(0, row) / L.entry(row, row));
        }
        
        // backward
        for (let row = rows - 1; row >= 0; --row) {
            for (let col = rows - 1; col > row; --col) {
                x.set_entry(0, row, x.entry(0, row) - U.entry(row, col) * x.entry(0, col));
            }
            x.set_entry(0, row, x.entry(0, row) / U.entry(row, row));
        }

        return x;
    }

    /**
     * {@link L}{@link U} decomposition of the Matrix {@link A}. Creates two matrices, so that the dot product LU equals A.
     * @param {Matrix} A 
     * @returns {{L: Matrix, U: Matrix}} result - Returns the left triangle matrix {@link L} and the upper triangle matrix {@link U}.
     */
    static LU(A) {
        let rows = A.shape[0];
        let L = new Matrix(rows, rows, "zeros");
        let U = new Matrix(rows, rows, "identity");
        let sum;

        for (let j = 0; j < rows; ++j) {
            for (let i = j; i < rows; ++i) {
                sum = 0;
                for (let k = 0; k < j; ++k) {
                    sum += L.entry(i, k) * U.entry(k, j);
                }
                /*sum = neumair_sum(linspace(0, j).map((k) => L.entry(i, k) * U.entry(k, j)))
                console.log("lsum1", sum)
                sum = []
                for (let k = 0; k < j; ++k) {
                    sum.push(L.entry(i, k) * U.entry(k, j))
                }
                sum = neumair_sum(sum)
                console.log("lsum2", sum)*/
                L.set_entry(i, j, A.entry(i, j) - sum);
            }
            for (let i = j; i < rows; ++i) {
                if (L.entry(j, j) === 0) {
                    return undefined;
                }
                sum = 0;
                for (let k = 0; k < j; ++k) {
                    sum += L.entry(j, k) * U.entry(k, i);
                }
                /*sum = neumair_sum(linspace(0, j).map((k) => L.entry(j, k) * U.entry(k, i)))
                console.log("usum1", sum)
                sum = []
                for (let k = 0; k < j; ++k) {
                    sum.push(L.entry(j, k) * U.entry(k, i))
                }
                sum = neumair_sum("usum2", sum)
                console.log(sum)*/
                U.set_entry(j, i, (A.entry(j, i) - sum) / L.entry(j, j));
            }
        }

        return { L: L, U: U };
    }

    /**
     * Computes the {@link k} components of the SVD decomposition of the matrix {@link M}
     * @param {Matrix} A 
     * @param {int} [k=2] 
     * @returns {{U: Matrix, Sigma: Matrix, V: Matrix}}
     */
    static SVD(A, k=2) {
        /*const MT = M.T;
        let MtM = MT.dot(M);
        let MMt = M.dot(MT);
        let { eigenvectors: V, eigenvalues: Sigma } = simultaneous_poweriteration(MtM, k);
        let { eigenvectors: U } = simultaneous_poweriteration(MMt, k);
        return { U: U, Sigma: Sigma.map(sigma => Math.sqrt(sigma)), V: V };*/
        
        //Algorithm 1a: Householder reduction to bidiagonal form:
        const [m, n] = A.shape;
        let U = new Matrix(m, n, (i, j) => i == j ? 1 : 0);
        console.log(U.to2dArray);
        let V = new Matrix(n, m, (i, j) => i == j ? 1 : 0);
        console.log(V.to2dArray);
        let B = Matrix.bidiagonal(A.clone(), U, V);
        console.log(U,V,B);
        return { U: U, "Sigma": B, V: V };
        
        
    }

}

/**
 * @class
 * @alias Hierarchical_Clustering
 */
class Hierarchical_Clustering {
    /**
     * @constructor
     * @memberof module:clustering
     * @alias Hierarchical_Clustering
     * @todo needs restructuring. 
     * @param {Matrix} matrix 
     * @param {("single"|"complete"|"average")} [linkage = "single"] 
     * @param {Function} [metric = euclidean] 
     * @returns {Hierarchical_Clustering}
     */
    constructor(matrix, linkage="single", metric=euclidean) {
        this._id = 0;
        this._matrix = matrix;
        this._metric = metric;
        this._linkage = linkage;
        this.init();
        this.root = this.do();
        return this;
    }

    /**
     * 
     * @param {Number} value - value where to cut the tree.
     * @param {("distance"|"depth")} [type = "distance"] - type of value.
     * @returns {Array<Array>} Array of clusters with the indices of the rows in given {@link matrix}.
     */
    get_clusters(value, type="distance") {
        let clusters = [];
        let accessor;
        switch (type) {
            case "distance":
                accessor = d => d.dist;
                break;
            case "depth":
                accessor = d => d.depth;
                break;
            default:
                throw "invalid type";
        }
        this._traverse(this.root, accessor, value, clusters);
        return clusters
    }

    /**
     * @private
     * @param {} node 
     * @param {*} f 
     * @param {*} value 
     * @param {*} result 
     */
    _traverse(node, f, value, result) {
        if (f(node) <= value) {
            result.push(node.leaves());
        } else {
            this._traverse(node.left, f, value, result);
            this._traverse(node.right, f, value, result);
        }
    }

    /**
     * computes the tree.
     */
    init() {
        const metric = this._metric;
        const A = this._matrix;
        const n = this._n = A.shape[0];
        const d_min = this._d_min = new Float64Array(n);
        const distance_matrix = this._distance_matrix = new Array(n);
        for (let i = 0; i < n; ++i) {
            d_min[i] = 0;
            distance_matrix[i] = new Float64Array(n);
            for (let j = 0; j < n; ++j) {
                distance_matrix[i][j] = i === j ? Infinity : metric(A.row(i), A.row(j));
                if (distance_matrix[i][d_min[i]] > distance_matrix[i][j]) {
                    d_min[i] = j;
                }
            }
        }
        const clusters = this._clusters = new Array(n);
        const c_size = this._c_size = new Uint16Array(n);
        for (let i = 0; i < n; ++i) {
            clusters[i] = [];
            clusters[i][0] = new Cluster(this._id++, null, null, 0, A.row(i), i, 1, 0);
            c_size[i] = 1;
        }
        return this;
    }

    /**
     * computes the tree.
     */
    do() {
        const n = this._n;
        const d_min = this._d_min;
        const D = this._distance_matrix;
        const clusters = this._clusters;
        const c_size = this._c_size;
        const linkage = this._linkage;
        let root = null;
        for (let p = 0, p_max = n - 1; p < p_max; ++p) {
            let c1 = 0;
            for (let i = 0; i < n; ++i) {
                if (D[i][d_min[i]] < D[c1][d_min[c1]]) {
                    c1 = i;
                }
            }
            let c2 = d_min[c1];
            let c1_cluster = clusters[c1][0];
            let c2_cluster = clusters[c2][0];
            let new_cluster = new Cluster(this._id++, c1_cluster, c2_cluster, D[c1][c2]);
            clusters[c1].unshift(new_cluster);
            c_size[c1] += c_size[c2];
            for (let j = 0; j < n; ++j) {
                switch(linkage) {
                    case "single":
                        if (D[c1][j] > D[c2][j]) {
                            D[j][c1] = D[c1][j] = D[c2][j];
                        }
                        break;
                    case "complete":
                        if (D[c1][j] < D[c2][j]) {
                            D[j][c1] = D[c1][j] = D[c2][j];
                        }
                        break;
                    case "average":
                        D[j][c1] = D[c1][j] = (c_size[c1] * D[c1][j] + c_size[c2] * D[c2][j]) / (c_size[c1] + c_size[j]);
                        break;
                }
            }
            D[c1][c1] = Infinity;
            for (let i = 0; i < n; ++i) {
                D[i][c2] = D[c2][i] = Infinity;
            }
            for (let j = 0; j < n; ++j) {
                if (d_min[j] === c2) {
                    d_min[j] = c1;
                }
                if (D[c1][j] < D[c1][d_min[c1]]) {
                    d_min[c1] = j;
                }
            }
            root = new_cluster;
        }
        return root;
    }
    
}

class Cluster {
    constructor(id, left, right, dist, centroid, index, size, depth) {
        this.id = id;
        this.left = left;
        this.right = right;
        this.dist = dist;
        this.index = index;
        this.size = size != null ? size : left.size + right.size;
        this.depth = depth != null ? depth : 1 + Math.max(left.depth, right.depth);
        this.centroid = centroid != null ? centroid : this._calculate_centroid(left, right);
        return this;
    }

    _calculate_centroid(left, right) {
        const l_size = left.size;
        const r_size = right.size;
        const l_centroid = left.centroid;
        const r_centroid = right.centroid;
        const size = this.size;
        const n = left.centroid.length;
        const new_centroid = new Float64Array(n);
        for (let i = 0; i < n; ++i) {
            new_centroid[i] = (l_size * l_centroid[i] + r_size * r_centroid[i]) / size;
        }
        return new_centroid;
    }

    get isLeaf() {
        return this.depth === 0;
    }

    leaves() {
        if (this.isLeaf) return [this.index];
        const left = this.left;
        const right = this.right;
        return (left.isLeaf ? [left.index] : left.leaves())
            .concat(right.isLeaf ? [right.index] : right.leaves())
    }
}

/**
 * @class
 * @alias Heap
 */
class Heap {
    /**
     * A heap is a datastructure holding its elements in a specific way, so that the top element would be the first entry of an ordered list.
     * @constructor
     * @memberof module:datastructure
     * @alias Heap
     * @param {Array=} elements - Contains the elements for the Heap. {@link elements} can be null.
     * @param {Function} [accessor = (d) => d] - Function returns the value of the element.
     * @param {("min"|"max"|Function)} [comparator = "min"] - Function returning true or false defining the wished order of the Heap, or String for predefined function. ("min" for a Min-Heap, "max" for a Max_heap)
     * @returns {Heap}
     * @see {@link https://en.wikipedia.org/wiki/Binary_heap}
     */
    constructor(elements = null, accessor = d => d, comparator = "min") {
        if (elements) {
            return Heap.heapify(elements, accessor, comparator);
        } else {
            this._accessor = accessor;
            this._container = [];
            if (comparator == "min") {
                this._comparator = (a, b) => a < b;
            } else if (comparator == "max") {
                this._comparator = (a, b) => a > b;
            } else {
                this._comparator = comparator;
            }
            return this
        }
    }

    /**
     * Creates a Heap from an Array
     * @param {Array|Set} elements - Contains the elements for the Heap.
     * @param {Function=} [accessor = (d) => d] - Function returns the value of the element.
     * @param {(String=|Function)} [comparator = "min"] - Function returning true or false defining the wished order of the Heap, or String for predefined function. ("min" for a Min-Heap, "max" for a Max_heap)
     * @returns {Heap}
     */
    static heapify(elements, accessor = d => d, comparator = "min") {
        const heap = new Heap(null, accessor, comparator);
        const container = heap._container;
        for (const e of elements) {
            container.push({
                "element": e,
                "value": accessor(e),
            });
        }
        for (let i = Math.floor((elements.length / 2) - 1); i >= 0; --i) {
            heap._heapify_down(i);
        }
        return heap;
    }

    /**
     * Swaps elements of container array.
     * @private
     * @param {Number} index_a 
     * @param {Number} index_b 
     */
    _swap(index_a, index_b) {
        const container = this._container;
        [container[index_b], container[index_a]] = [container[index_a], container[index_b]];
        return;
    }

    /**
     * @private
     */
    _heapify_up() {
        const container = this._container;
        let index = container.length - 1;
        while (index > 0) {
            let parentIndex = Math.floor((index - 1) / 2);
            if (!this._comparator(container[index].value, container[parentIndex].value)) {
                break;
            } else {
            this._swap(parentIndex, index);
            index = parentIndex;
            }
        }
    }

    /**
     * Pushes the element to the heap.
     * @param {} element
     * @returns {Heap}
     */
    push(element) {
        const value = this._accessor(element);
        //const node = new Node(element, value);
        const node = {"element": element, "value": value};
        this._container.push(node);
        this._heapify_up();
        return this;
    }

    /**
     * @private
     * @param {Number} [start_index = 0] 
     */
    _heapify_down(start_index=0) {
        const container = this._container;
        const comparator = this._comparator;
        const length = container.length;
        let left = 2 * start_index + 1;
        let right = 2 * start_index + 2;
        let index = start_index;
        if (index > length) throw "index higher than length"
        if (left < length && comparator(container[left].value, container[index].value)) {
            index = left;
        }
        if (right < length && comparator(container[right].value, container[index].value)) {
            index = right;
        }
        if (index !== start_index) {
            this._swap(start_index, index);
            this._heapify_down(index);
        }
    }

    /**
     * Removes and returns the top entry of the heap.
     * @returns {Object} Object consists of the element and its value (computed by {@link accessor}).
     */
    pop() {
        const container = this._container;
        if (container.length === 0) {
            return null;
        } else if (container.length === 1) {
            return container.pop();
        }
        this._swap(0, container.length - 1);
        const item = container.pop();
        this._heapify_down();
        return item;
    }

    /**
     * Returns the top entry of the heap without removing it.
     * @returns {Object} Object consists of the element and its value (computed by {@link accessor}).
     */
    get first() {
        return this._container.length > 0 ? this._container[0] : null;
    }


    /**
     * Yields the raw data
     * @yields {Object} Object consists of the element and its value (computed by {@link accessor}).
     */
    * iterate() {
        for (let i = 0, n = this._container.length; i < n; ++i) {
            yield this._container[i].element;
        }
    }

    /**
     * Returns the heap as ordered array.
     * @returns {Array} Array consisting the elements ordered by {@link comparator}.
     */
    toArray() {
        return this.data()
            .sort((a,b) => this._comparator(a, b) ? -1 : 0)
    }

    /**
     * Returns elements of container array.
     * @returns {Array} Array consisting the elements.
     */
    data() {
        return this._container
            .map(d => d.element)
    }

    /**
     * Returns the container array.
     * @returns {Array} The container array.
     */
    raw_data() {
        return this._container;
    }

    /**
     * The size of the heap.
     * @returns {Number}
     */
    get length() {
        return this._container.length;
    }

    /**
     * Returns false if the the heap has entries, true if the heap has no entries.
     * @returns {Boolean}
     */
    get empty() {
        return this.length === 0;
    }
}

/**
 * @module datastructure
 */

/**
 * @class
 * @alias KMeans
 */
class KMeans {
    /**
     * @constructor
     * @memberof module:clustering
     * @alias KMeans
     * @todo needs restructuring. 
     * @param {Matrix} matrix 
     * @param {Numbers} K 
     * @param {Function} [metric = euclidean] 
     * @param {Number} [seed = 1987]
     * @param {Boolean} [init = true]
     * @returns {KMeans}
     */
    constructor(matrix, K, metric = euclidean, seed=1987, init = true) {
        this._metric = metric;
        this._matrix = matrix;
        this._K = K;
        const [N, D] = matrix.shape;
        this._N = N;
        this._D = D;
        if (K > N) K = N;
        this._randomizer = new Randomizer(seed);
        this._clusters = new Array(N).fill(undefined);
        this._cluster_centroids = this._get_random_centroids(K);
        if (init) this.init(K, this._cluster_centroids);
        return this;
    }

    /**
     * @returns {Array<Array>} - Array of clusters with the indices of the rows in given {@link matrix}. 
     */
    get_clusters() {
        const K = this._K;
        const clusters = this._clusters;
        const result = new Array(K).fill().map(() => new Array());
        clusters.forEach((c, i) => result[c].push(i));
        return result;
    }

    /**
     * @private
     * @param {Array} points 
     * @param {Array} candidates 
     */
    _furthest_point(points, candidates) {
        const A = this._matrix;
        const metric = this._metric;
        let i = points.length;
        let H = Heap.heapify(
            candidates, 
            (d) => {
                const Ad = A.row(d);
                let sum = 0;
                for (let j = 0; j < i; ++j) {
                    sum += metric(Ad, points[j]);
                }
                return sum;
            }, 
            "max"
        );
        return H.pop().element;
    }

    _get_random_centroids(K) {
        const N = this._N;
        const randomizer = this._randomizer;
        const A = this._matrix;
        const cluster_centroids = new Array(K).fill();
        const indices = linspace(0, N - 1);
        const random_point = randomizer.random_int % (N - 1);
        cluster_centroids[0] = A.row(random_point);
        const init_points = [random_point];
        const sample_size = Math.floor((N - K) / K);// / K
        for (let i = 1; i < K; ++i) {
            // sampling + kmeans++ improvement?
            const sample = randomizer.choice(indices.filter(d => init_points.indexOf(d) == -1), sample_size);
            const furthest_point = this._furthest_point(cluster_centroids.slice(0, i), sample);
            init_points.push(furthest_point);
            cluster_centroids[i] = A.row(furthest_point);
        }
        return cluster_centroids;
    }

    _iteration(cluster_centroids) {
        const K = cluster_centroids.length;
        const N = this._N;
        const D = this._D;
        const A = this._matrix;
        const metric = this._metric;
        const clusters = this._clusters;
        let clusters_changed = false;
        // find nearest cluster centroid.
        for (let i = 0; i < N; ++i) {
            const Ai = A.row(i);
            let min_dist = Infinity;
            let min_cluster = null;
            for (let j = 0; j < K; ++j) {
                let d = metric(cluster_centroids[j], Ai);
                if (d < min_dist) {
                    min_dist = d;
                    min_cluster = j; 
                }
            }
            if (clusters[i] !== min_cluster) {
                clusters_changed = true;
            }
            clusters[i] = min_cluster;
        }
        // update cluster centroid
        // reset cluster centroids to 0
        for (let i = 0; i < K; ++i) {
            const centroid = cluster_centroids[i];
            for (let j = 0; j < D; ++j) {
                centroid[j] = 0;
            }
        }
        // compute centroid
        const cluster_counter = new Array(K).fill(0);
        for (let i = 0; i < N; ++i) {
            const Ai = A.row(i);
            const ci = clusters[i];
            cluster_counter[ci]++;
            const centroid = cluster_centroids[ci];
            for (let j = 0; j < D; ++j) {
                centroid[j] += Ai[j];
            }
        }
        for (let i = 0; i < K; ++i) {
            const n = cluster_counter[i];
            cluster_centroids[i] = cluster_centroids[i].map(c => {
                return c / n;
            });
        }
        return {   
            "clusters_changed": clusters_changed,
            "cluster_centroids": cluster_centroids
        };
    }

    /**
     * Computes {@link K} clusters out of the {@link matrix}.
     * @param {Number} K - number of clusters.
     */
    init(K, cluster_centroids) {
        if (!K) K = this._K;
        if (!cluster_centroids) cluster_centroids = this._get_random_centroids(K);
        let clusters_changed = false;
        do {
            const iteration_result = this._iteration(cluster_centroids);
            cluster_centroids = iteration_result.cluster_centroids;
            clusters_changed = iteration_result.clusters_changed;
        } while (clusters_changed)
    }
    
}

/**
 * @class
 * @alias XMeans
 */
class XMeans{
    /**
     * @constructor
     * @memberof module:clustering
     * @alias XMeans
     * @todo needs restructuring and repairing!!
     * @param {Matrix} matrix 
     * @param {Numbers} K_max
     * @param {Numbers} K_min
     * @param {Function} [metric = euclidean] 
     * @param {Number} [seed = 1987]
     * @returns {XMeans}
     * @see {@link https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf}
     * @see {@link https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/xmeans.py}
     * @see {@link https://github.com/haifengl/smile/blob/master/core/src/main/java/smile/clustering/XMeans.java}
     */
    constructor(matrix, K_max = 10, K_min = 2, metric = euclidean, seed=1987) {
        //const first = super(matrix, K_min, metric, seed, false);
        const first = new KMeans(matrix, K_min, metric, seed, false);
        this._K_max = K_max;
        this._K_min = K_min;
        this._metric = metric;
        first.init(K_min, first._get_random_centroids(K_min));
        const randomizer = this._randomizer = first._randomizer;
        const centroids = first._cluster_centroids;
        const candidates = this._candidates = {};
        let K = K_min;
        candidates[K] = {
            "kmeans": first,
            "cluster_centroids": centroids,
            "score": null,
        };
        const A = this._matrix = matrix;
        const N = A.shape[0];
        // foreach K in [K_min, K_max];
        do {
            console.log(K, candidates);
            const candidate = candidates[K];
            const clusters = candidate.kmeans.get_clusters();
            const parent_bic = this._bic(clusters, centroids, linspace(0, N - 1));
            candidate.score = parent_bic;
            const child_bic = [];
            const child_kmeans = [];
            const child_indices = [];
            // foreach cluster
            for (let j = 0; j < K; ++j) {
                const cluster = clusters[j];
                console.log(cluster.length);
                if (cluster.length < K_max) continue;
                const subset = Matrix.from(cluster.map(d => A.row(d)));
                const subset_kmeans = new KMeans(subset, 2, metric, 1987, false);
                subset_kmeans._randomizer = randomizer;
                subset_kmeans.init();
                const subset_cluster = subset_kmeans.get_clusters();
                const subset_centroids = subset_kmeans._cluster_centroids;
                const bic = this._bic(subset_cluster, subset_centroids, cluster);
                child_bic.push(bic);
                child_kmeans.push(subset_kmeans);
                child_indices.push(j);
            }
            //if (child_bic.length < (K )) break;
            //choose best
            let best_split = child_indices[0];
            let best_bic = child_bic[0];
            for (let i = 0; i < child_bic.length; ++i) {
                if (best_bic > child_bic[i]) {
                    best_split = child_indices[i];
                    best_bic = child_bic[i];
                }
            }
            const best_cluster_centroids = candidate.cluster_centroids.splice(best_split, 1, ...child_kmeans[best_split]._cluster_centroids);
            console.log(candidate.cluster_centroids, child_kmeans[best_split]._cluster_centroids);
            
            const parent_clusters = candidate.kmeans._clusters;
            const best_candidate_clusters = child_kmeans[best_split]._clusters;
            const best_candidate = new KMeans(A, K + 1, metric, 1987, false);
            best_candidate._randomizer = randomizer;
            // set clusters and centroids
            let counter = 0;
            //let cluster_number = best_cluster_centroids.length;
            best_candidate._clusters = parent_clusters.map(c => {
                if (c == best_split) {
                    return c + best_candidate_clusters[counter++];
                } else if (c > best_split) {
                    return c + 1;
                }
                return c;
            });
            //best_candidate._K = K + 1;
            console.log(best_candidate.get_clusters());
            //best_candidate.init(K + 1, best_cluster_centroids);
            console.log(best_candidate);
            //save best candidate.
            candidates[K + 1] = {
                "kmeans": best_candidate,
                "cluster_centroids": best_cluster_centroids,
                "score": child_bic[best_split],
            };
            

        } while (++K < K_max)


        // return best candidate.
        return this;
    }

    get_clusters() {
        let K_min = this._K_min;
        let K_max = this._K_max;
        const candidates = this._candidates;
        let best_score = candidates[K_min].score;
        let best_candidate = candidates[K_min].kmeans;
        for (let i = K_min + 1; i < K_max; ++i) {
            if (!(i in candidates)) break;
            const candidate = candidates[i];
            const score = candidate.score;
            if (best_score < score) {
                best_score = score;
                best_candidate = candidate.kmeans;
            }
        }
        return best_candidate.get_clusters();
    }
    
    _bic(clusters, centroids, indices) {
        const A = this._matrix;
        const D = this._matrix.shape[1];
        const K = centroids.length;
        //const result = new Array(K).fill();
        let result = 0;

        let variance = 0;
        for (let i = 0; i < K; ++i) {
            const cluster = clusters[i];
            const centroid = centroids[i];
            const n = cluster.length;
            for (let j = 0; j < n; ++j) {
                variance += euclidean_squared(centroid, A.row(indices[cluster[j]])) ** 2;
            }
        }
        const N = clusters.reduce((a, b) => a + b.length, 0);
        const p = (K - 1) + D * K + 1;
        variance /= (N - K);

        for (let i = 0; i < K; ++i) {
            const n = clusters[i].length;
            const log_likelihood = 
                (n * Math.log(2 * Math.PI) -
                n * D * Math.log(variance) -
                (n - K)) * .5 + 
                n * Math.log(n) - 
                n * Math.log(N);
            result += log_likelihood - p * .5 * Math.log(N);
        }
        return result;
    }
    
}

/**
 * @class
 * @alias OPTICS
 */
class OPTICS {
    /**
     * **O**rdering **P**oints **T**o **I**dentify the **C**lustering **S**tructure.
     * @constructor
     * @memberof module:clustering
     * @alias OPTICS
     * @todo needs restructuring. 
     * @param {Matrix} matrix - the data.
     * @param {Number} epsilon - the minimum distance which defines whether a point is a neighbor or not.
     * @param {Number} min_points - the minimum number of points which a point needs to create a cluster. (Should be higher than 1, else each point creates a cluster.)
     * @param {Function} [metric = euclidean] - the distance metric which defines the distance between two points of the {@link matrix}.
     * @returns {OPTICS}
     * @see {@link https://www.dbs.ifi.lmu.de/Publikationen/Papers/OPTICS.pdf}
     * @see {@link https://en.wikipedia.org/wiki/OPTICS_algorithm}
     */
    constructor(matrix, epsilon, min_points, metric = euclidean) {
        this._matrix = matrix;
        this._epsilon = epsilon;
        this._min_points = min_points;
        this._metric = metric;

        this._ordered_list = [];
        this._clusters = [];
        this._DB = new Array(matrix.shape[0]).fill();
        this.init();
        return this;
    }

    /**
     * Computes the clustering.
     */
    init() {
        const ordered_list = this._ordered_list;
        const matrix = this._matrix;
        const N = matrix.shape[0];
        const DB = this._DB;
        const clusters = this._clusters;
        let cluster_index = this._cluster_index = 0;

        for (let i = 0; i < N; ++i) {
            DB[i] = {
                "element": matrix.row(i),
                "index": i,
                "reachability_distance": undefined,
                "processed": false,
            };
        }
        for (const p of DB) {
            if (p.processed) continue;
            p.neighbors = this._get_neighbors(p);
            p.processed = true;
            clusters.push([p.index]);
            cluster_index = clusters.length - 1;
            ordered_list.push(p);
            if (this._core_distance(p) != undefined) {
                const seeds = new Heap(null, d => d.reachability_distance, "min");
                this._update(p, seeds);
                this._expand_cluster(seeds, clusters[cluster_index]);
            }
        }
    }

    /**
     * 
     * @private
     * @param {Object} p - a point of {@link matrix}.
     * @returns {Array} An array consisting of the {@link epsilon}-neighborhood of {@link p}.
     */
    _get_neighbors(p) {
        if ("neighbors" in p) return p.neighbors;
        const DB = this._DB;
        const metric = this._metric;
        const epsilon = this._epsilon;
        const neighbors = [];
        for (const q of DB) {
            if (q.index == p.index) continue;
            if (metric(p.element, q.element) < epsilon) {
                neighbors.push(q);
            }
        }
        return neighbors;
    }

    /**
     * 
     * @private
     * @param {Object} p - a point of {@link matrix}.
     * @returns {Number} The distance to the {@link min_points}-th nearest point of {@link p}, or undefined if the {@link epsilon}-neighborhood has fewer elements than {@link min_points}.
     */
    _core_distance(p) {
        const min_points = this._min_points;
        const metric = this._metric;
        if (p.neighbors && p.neighbors.length <= min_points) {
            return undefined;
        }
        return metric(p.element, p.neighbors[min_points].element);
    }

    /**
     * Updates the reachability distance of the points.
     * @private
     * @param {Object} p 
     * @param {Heap} seeds 
     */
    _update(p, seeds) {
        const metric = this._metric;
        const core_distance = this._core_distance(p);
        const neighbors = this._get_neighbors(p);//p.neighbors;
        for (const q of neighbors) {
            if (q.processed) continue;
            const new_reachability_distance = Math.max(core_distance, metric(p.element, q.element));
            //if (q.reachability_distance == undefined) { // q is not in seeds
            if (seeds.raw_data().findIndex(d => d.element == q) < 0) {
                q.reachability_distance = new_reachability_distance;
                seeds.push(q);
            } else { // q is in seeds
                if (new_reachability_distance < q.reachability_distance) {
                    q.reachability_distance = new_reachability_distance;
                    seeds = Heap.heapify(seeds.data(), d => d.reachability_distance, "min"); // seeds change key =/
                }
            }
        }
    }

    /**
     * Expands the {@link cluster} with points in {@link seeds}.
     * @private
     * @param {Heap} seeds 
     * @param {Array} cluster 
     */
    _expand_cluster(seeds, cluster) {
        const ordered_list = this._ordered_list;
        while (!seeds.empty) {
            const q = seeds.pop().element;
            q.neighbors = this._get_neighbors(q);
            q.processed = true;
            cluster.push(q.index);
            ordered_list.push(q);
            if (this._core_distance(q) != undefined) {
                this._update(q, seeds);
                this._expand_cluster(seeds, cluster);
            }
        }
    }

    /**
     * Returns an array of clusters.
     * @returns {Array<Array>} Array of clusters with the indices of the rows in given {@link matrix}.
     */
    get_clusters() {
        const clusters = [];
        const outliers = [];
        const min_points = this._min_points;
        for (const cluster of this._clusters) {
            if (cluster.length < min_points) {
                outliers.push(...cluster);
            } else {
                clusters.push(cluster);
            }
        }
        clusters.push(outliers);
        return clusters;
    }

    /**
     * @returns {Array} Returns an array, where the ith entry defines the cluster affirmation of the ith point of {@link matrix}. (-1 stands for outlier)
     */
    get_cluster_affirmation() {
        const N = this._matrix.shape[0];
        const result = new Array(N).fill();
        const clusters = this.get_clusters();
        for (let i = 0, n = clusters.length; i < n; ++i) {
            const cluster = clusters[i];
            for (const index of cluster) {
                result[index] = (i < n - 1) ? i : -1;
            }
        }
        return result;
    }
}

/**
 * @module clustering
 */

/**
 * @class
 * @alias HNSW
 */
class HNSW {
    /**
     * Hierarchical navigable small world graph. Efficient and robust approximate nearest neighbor search.
     * @constructor
     * @memberof module:knn
     * @alias HNSW
     * @param {Function} [metric = euclidean] - metric to use: (a, b) => distance.
     * @param {Boolean} [heuristic = true] - use heuristics or naive selection.
     * @param {Number} [m = 5] - max number of connections.
     * @param {Number} [ef = 200] - size of candidate list.
     * @param {Number} [m0 = 2 * m] - max number of connections for ground layer.
     * @param {Number} [mL = 1 / Math.log2(m)] - normalization factor for level generation.
     * @param {Number} [seed = 1987] - seed for random number generator.
     * @see {@link https://arxiv.org/abs/1603.09320}
     * @see {@link https://arxiv.org/pdf/1904.02077}
     */
    constructor(metric = euclidean, heuristic = true, m = 5, ef = 200, m0 = null, mL = null, seed = 1987) {
        this._metric = metric;
        this._select = heuristic ? this._select_heuristic : this._select_simple;
        this._m = m;
        this._ef = ef;
        this._m0 = m0 || 2 * m;
        this._graph = new Map();
        this._ep = null;
        this._L = null;
        this._mL = mL || 1 / Math.log2(m);
        this._randomizer = new Randomizer(seed);
    }

    addOne(element) {
        this.add([element]);
    }

    /**
     * 
     * @param {Array<*>} elements - new elements.
     * @returns {HNSW}
     */
    add(elements) {
        const m = this._m;
        const ef = this._ef;
        const m0 = this._m0;
        const mL = this._mL;
        const randomizer = this._randomizer;
        let graph = this._graph;
        for (const element of elements) {
            let ep = this._ep ? this._ep.slice() : null;
            let W = [];
            const L = this._L;
            const rand = Math.min(randomizer.random + 1e-8, 1);
            let l = Math.floor(-Math.log(rand * mL));
            const min_L_l = Math.min(L, l);
            if (L) {
                for (let l_c = graph.size - 1; l_c > min_L_l; --l_c) {
                    ep = this._search_layer(element, ep, 1, l_c);
                }
                for (let l_c = min_L_l; l_c >= 0; --l_c) {
                    const layer_c = graph.get(l_c);//[l_c];
                    layer_c.points.push(element);
                    W = this._search_layer(element, ep, ef, l_c);
                    const neighbors = l_c > 3 ? this._select(element, W, m, l_c) : this._select_simple(element, W, m);
                    for (const p of neighbors) {
                        if (p !== element) {
                            const edges_p = layer_c.edges.get(p);
                            if (!edges_p) {
                                layer_c.edges.set(p, [element]) ;
                            }else {
                                edges_p.push(element);
                            }
                            const edges_e = layer_c.edges.get(element);
                            if (!edges_e) {
                                layer_c.edges.set(element, [p]) ;
                            } else {
                                edges_e.push(p);
                            }
                        }
                    }
                    const max = (l_c === 0 ? m0 : m);
                    for (const e of neighbors) {
                        const e_conn = layer_c.edges.get(e);
                        if (e_conn.length > max) {
                            const neighborhood = this._select(e, e_conn, max, l_c);
                            layer_c.edges.delete(e);
                            layer_c.edges.set(e, neighborhood);
                        }
                    }
                    ep = W;
                }
            }
            let N = graph.size;
            if (N < l || l > L) {
                for (let i = N; i <= l; ++i) {
                    graph.set(i, {
                        "l_c": i, 
                        "points": [element], 
                        "edges": new Map(),
                    });
                }
                this._ep = [element];
                this._L = l;
            }
        }
        return this;
    }

    /**
     * @private
     * @param {*} q - base element.
     * @param {Array} candidates - candidate elements.
     * @param {Number} M - number of neighbors to return.
     * @param {Number} l_c - layer number.
     * @param {Boolean} [extend_candidates = true] - flag indicating wheter or not to extend candidate list.
     * @param {Boolean} [keep_pruned_connections = true] - flag indicating wheter or not to add discarded elements.
     * @returns M elements selected by the heuristic.
     */
    _select_heuristic(q, candidates, M, l_c, extend_candidates = true, keep_pruned_connections = true) {
        if (l_c > this._graph.size - 1) return candidates
        const metric = this._metric;
        const randomizer = this._randomizer;
        const layer = this._graph.get(l_c);
        let R = [];
        let W_set = new Set(candidates);
        if (extend_candidates) {
            for (const c of candidates) {
                const edges = layer.edges.get(c);
                if (!edges) break;
                for (const c_adj of edges) {
                    W_set.add(c_adj);
                }
            }
        }
        let W = new Heap(W_set, d => metric(d, q), "min");
        let W_d = new Heap(null, d => metric(d, q), "min");
        while (!W.empty && R.length < M) {
            let e = W.pop();
            let random_r = randomizer.random_int % R.length;
            if (R.length === 0 || e.value < metric(R[random_r], q)) {
                R.push(e.element);
            } else {
                W_d.push(e.element);
            }
        }
        if (keep_pruned_connections) {
            while (!W_d.empty && R.length < M) {
                R.push(W_d.pop().element);
            }
        }
        return R
    }

    /**
     * @private
     * @param {*} q - base element.
     * @param {Array} C - candidate elements.
     * @param {Number} M - number of neighbors to return.
     * @returns {Array} M nearest elements from C to q.
     */
    _select_simple(q, C, M) {
        const metric = this._metric;
        let res = C.sort((a,b) => metric(a, q) - metric(b, q)).slice(0,M);
        return res
    }

    /**
     * @private
     * @param {*} q - query element.
     * @param {Array} ep - enter points.
     * @param {Number} ef - number of nearest to {@link q} elements to return.
     * @param {Number} l_c - layer number.
     * @returns {Array} ef closest neighbors to q.
     */
    _search_layer(q, ep, ef, l_c) {
        const metric = this._metric;
        const layer = this._graph.get(l_c);//.find(l => l.l_c === l_c);//[l_c];
        if (layer.edges.size === 0) return ep;
        let v = new Set(ep);
        let C = new Heap(v, d => metric(d, q), "min");
        let W = new Heap(v, d => metric(d, q), "max");
        while (!C.empty) {
            const c = C.pop();
            let f = W.first;
            if (c.value > f.value) {
                break;
            }
            const edges = layer.edges.get(c.element);
            if (!edges) break;
            for (const e of edges) {
                if (!v.has(e)) {
                    v.add(e);
                    f = W.first;
                    if (metric(e, q) < metric(f.element, q) || W.length < ef) {
                        C.push(e);
                        W.push(e);
                        if (W.length > ef) {
                            W.pop();
                        }
                    }
                }
            }
        }
        return W.data();
    }

    /**
     * 
     * @param {*} q - query element.
     * @param {*} K - number of nearest neighbors to return.
     * @param {*} ef - size of the dynamic cnadidate list.
     * @returns {Array} K nearest elements to q.
     */
    search(q, K, ef = 1) {
        let ep = this._ep.slice();
        let L = this._L;
        for (let l_c = L; l_c > 0; --l_c) {
            ep = this._search_layer(q, ep, ef, l_c);
        }
        ep = this._search_layer(q, ep, K, 0);
        return ep;
    }

    /**
     * Iterator for searching the HNSW graphs
     * @param {*} q - query element.
     * @param {*} K - number of nearest neighbors to return.
     * @param {*} ef - size of the dynamic cnadidate list.
     * @yields {Array} K nearest elements to q.
     */
    * search_iter(q, K, ef = 1) {
        let ep = this._ep.slice();
        let L = this._L;
        yield {"l_c": L, "ep": [q]};
        for (let l_c = L; l_c > 0; --l_c) {
            yield {"l_c": l_c, "ep": ep};
            ep = this._search_layer(q, ep, ef, l_c);
            yield {"l_c": l_c, "ep": ep};
        }
        yield {"l_c": 0, "ep": ep};
        ep = this._search_layer(q, ep, K, 0);
        yield {"l_c": 0, "ep": ep};
    }
}

/**
 * @class
 * @alias BallTree
 */
class BallTree {
    /**
     * Generates a BallTree with given {@link elements}.
     * @constructor
     * @memberof module:knn
     * @alias BallTree
     * @param {Array=} elements - Elements which should be added to the BallTree
     * @param {Function} [metric = euclidean] metric to use: (a, b) => distance
     * @see {@link https://en.wikipedia.org/wiki/Ball_tree}
     * @see {@link https://github.com/invisal/noobjs/blob/master/src/tree/BallTree.js}
     * @returns {BallTree}
     */
    constructor(elements = null, metric = euclidean) {
        this._Node = class {
            constructor(pivot, child1=null, child2=null, radius=null) {
                this.pivot = pivot;
                this.child1 = child1;
                this.child2 = child2;
                this.radius = radius;
            }
        };
        this._Leaf = class {
            constructor(points) {
                this.points = points;
            }
        };
        this._metric = metric;
        if (elements) {
            this.add(elements);
        }
        return this;
    }

    /**
     * 
     * @param {Array<*>} elements - new elements.
     * @returns {BallTree}
     */
    add(elements) {
        elements = elements.map((element, index) => {
            return {index: index, element: element}
        });
        this._root = this._construct(elements);
        return this;
    }

    /**
     * @private
     * @param {Array<*>} elements 
     * @returns {Node} root of balltree.
     */
    _construct(elements) {
        if (elements.length === 1) {
            return new this._Leaf(elements);
        } else {
            let c = this._greatest_spread(elements);
            let sorted_elements = elements.sort((a, b) => a.element[c] - b.element[c]);
            let n = sorted_elements.length;
            let p_index = Math.floor(n / 2);
            let p = elements[p_index];
            let L = sorted_elements.slice(0, p_index);
            let R = sorted_elements.slice(p_index, n);
            let radius = Math.max(...elements.map(d => this._metric(p.element, d.element)));
            let B;
            if (L.length > 0 && R.length > 0) {         
                B = new this._Node(p, this._construct(L), this._construct(R), radius);
            } else {
                B = new this._Leaf(elements);
            }
            return B;
        }
    }

    /**
     * @private
     * @param {Node} B 
     * @returns {Number}
     */
    _greatest_spread(B) {
        let d = B[0].element.length;
        let start = new Array(d);

        for (let i = 0; i < d; ++i) {
            start[i] = [Infinity, -Infinity];
        }

        let spread = B.reduce((acc, current) => {
            for (let i = 0; i < d; ++i) {
                acc[i][0] = Math.min(acc[i][0], current.element[i]);
                acc[i][1] = Math.max(acc[i][1], current.element[i]);
            }
            return acc;
        }, start);
        spread = spread.map(d => d[1] - d[0]);
        
        let c = 0;
        for (let i = 0; i < d; ++i) {
            c = spread[i] > spread[c] ? i : c;
        }
        return c
    }

    /**
     * 
     * @param {*} t - query element.
     * @param {Number} [k = 5] - number of nearest neighbors to return.
     * @returns {Heap} - Heap consists of the {@link k} nearest neighbors.
     */
    search(t, k = 5) {
        return this._search(t, k, new Heap(null, d => this._metric(d.element, t), "max"), this._root);
    }

    /**
     * @private
     * @param {*} t - query element.
     * @param {Number} [k = 5] - number of nearest neighbors to return.
     * @param {Heap} Q - Heap consists of the currently found {@link k} nearest neighbors.
     * @param {Node|Leaf} B 
     */
    _search(t, k, Q, B) {
        // B is Node
        if (Q.length >= k && B.pivot && B.radius && this._metric(t, B.pivot.element) - B.radius >= Q.first.value) {
            return Q;
        } 
        if (B.child1) this._search(t, k, Q, B.child1);
        if (B.child2) this._search(t, k, Q, B.child2);
        
        // B is leaf
        if (B.points) {
            for (let i = 0, n = B.points.length; i < n; ++i) {
                let p = B.points[i];
                if (k > Q.length) {
                    Q.push(p);
                } else {
                    Q.push(p);
                    Q.pop();
                }
            }
        }
        return Q;
    }


}

/**
 * @class
 * @alias NNDescent
 */
class NNDescent{
    /**
     * @constructor
     * @memberof module:knn
     * @alias NNDescent
     * @param {Array<*>=} elements - called V in paper.
     * @param {Function} [metric = euclidean] - called sigma in paper.
     * @param {Number} [K = 10] - number of neighbors {@link search} should return.
     * @param {Number} [rho = .8] - sample rate.
     * @param {Number} [delta = 0.0001] - precision parameter.
     * @param {Number} [seed = 1987] - seed for the random number generator.
     * @returns {NNDescent}
     * @see {@link http://www.cs.princeton.edu/cass/papers/www11.pdf}
     */
    constructor(elements, metric=euclidean, K = 10, rho = 1, delta = 1e-3, seed=19870307) {
        this._metric = metric;
        this._randomizer = new Randomizer(seed);
        this._K = K;
        this._rho = rho;
        this._sample_size = K * rho;
        this._delta = delta;
        if (elements) {
            this.add(elements);
        }
        return this;   
    }

    /**
     * Samples Array A with sample size.
     * @private
     * @param {Array<*>} A 
     */
    _sample(A) {
        const n = A.length;
        const sample_size = this._sample_size;
        if (sample_size > n) {
            return A;
        } else {
            const randomizer = this._randomizer;
            return randomizer.choice(A, sample_size);
        }
    }

    /**
     * Updates the KNN heap and returns 1 if changed, or 0 if not.
     * @private
     * @param {KNNHeap} B 
     * @param {*} u 
     */
    _update(B, u) {
        if (B.push(u)) {
            u.flag = true;
            B.pop();
            return 1;
        } else {
            return 0;
        }
    }

    /**
     * Collects for each element where it is neighbor from.
     * @private
     * @param {Array<KNNHeap>} B 
     */
    _reverse(B) {
        const N = this._N;
        const R = new Array(N).fill().map(() => new Array());
        for (let i = 0; i < N; ++i) {
            for (let j = 0; j < N; ++j) {
                const Bjdata = B[j].data();
                const val = Bjdata.find(d => d.index === i);
                if (val) R[j].push(val);
            }
        }
        return R;
    }

    /**
     * 
     * @param {Array} elements 
     */
    add(elements) {
        this._elements = elements = elements.map((e, i) => {
            return {
                "element": e,
                "index": i,
                "flag": true,
            }
        });
        const randomizer = this._randomizer;
        const metric = this._metric;
        const K = this._K;
        const delta = this._delta;
        const N = this._N = elements.length;
        const B = this._B = new Array();
        // B[v] <-- Sample(V,K)
        for (let i = 0; i < N; ++i) {
            const e = elements[i];
            const sample = randomizer.choice(elements, K);
            const Bi = new KNNHeap(sample, (d) => metric(d.element, e.element), "max"); // "max" to pop the futherst elements away
            B.push(Bi);
        }

        // loop
        let c = Infinity;
        let old_c = -Infinity;
        //let min_iter = 10;
        //let max_iter = 20;
        //while (min_iter-- > 0 || (c < delta * N * K) && max_iter-- > 0) {
        while (c > (delta * N * K) && c != old_c) {
            // parallel for v e V do
            const old_ = new Array(N);
            const new_ = new Array(N);
            for (let i = 0; i < N; ++i) {
                const e = elements[i];
                const Bi = B[i].data();
                const falseBs = Bi.filter(d => !d.flag);
                const trueBs = this._sample(Bi.filter(d => d.flag));
                trueBs.forEach(d => d.flag = false);
                old_[i] = new KNNHeap(falseBs, (d) => metric(d.element, e.element), "max");
                new_[i] = new KNNHeap(trueBs, (d) => metric(d.element, e.element), "max");
            }
            const old_reverse = this._reverse(old_);
            const new_reverse = this._reverse(new_);
            old_c = c;
            c = 0;
            // parallel for v e V do
            for (let i = 0; i < N; ++i) {
                this._sample(old_reverse[i]).forEach(o => old_[i].push(o));
                this._sample(new_reverse[i]).forEach(n => new_[i].push(n));
                const new_i = new_[i].data();
                const old_i = old_[i].data();
                const n1 = new_i.length;
                const n2 = old_i.length;
                for (let j = 0; j < n1; ++j) {
                    const u1 = new_i[j];
                    const Bu1 = B[u1.index];
                    for (let k = 0; k < n1; ++k) {
                        const u2 = new_i[k];
                        if (u1 == u2) continue;
                        const Bu2 = B[u2.index];
                        c += this._update(Bu2, u1);
                        c += this._update(Bu1, u2);
                    }
                    for (let k = 0; k < n2; ++k) {
                        const u2 = old_i[k];
                        if (u1 == u2) continue;
                        const Bu2 = B[u2.index];
                        c += this._update(Bu2, u1);
                        c += this._update(Bu1, u2);
                    }
                }
            }
        } 
        return this;
    }

    /**
     * @todo not implemented yet
     * @param {*} x 
     * @param {*} k 
     */
    search(x, k=5) {
        return this._B[this._randomizer.random_int % (this._N - 1)].toArray().slice(0, k);
    }

    search_index(i, k=5) {
        const B = this._B[i];
        const result = B.raw_data().sort((a, b) => a.value - b.value).slice(-k);
        return result;
    }
}

class KNNHeap extends Heap{
    constructor(elements, accessor, comparator) {
        super(null, accessor, comparator);
        this.set = new Set();
        if (elements) {
            for (const element of elements) {
                this.push(element);
            }
        }
    }

    push(element) {
        const set = this.set;
        if (set.has(element)){
            return false;
        } else {
            set.add(element);
            super.push(element);
            return true;
        }    
    }

    pop() {
        super.pop().element;
        //const element = super.pop().element;
        // once popped it should not return into the heap.
        // used as max heap. therefore, if popped the furthest 
        // element in the knn list gets removed.
        // this.set.delete(element); 
    }
}

/**
 * @module knn
 */

// https://github.com/jdfekete/reorder.js/blob/master/reorder.v1.js
class Reorder{
    constructor(A) {
        this._A = A;
        this._optimal_leaf_order = null;
    }

    get available() {
        return [
            "optimal_leaf_order",
            "spectral_order",
        ]
    }

    reorder(type="optimal_leaf_order", metric=euclidean) {
        let result = null;
        switch (type) {
            case "optimal_leaf_order":
                if (!this._optimal_leaf_order) 
                    this._optimal_leaf_order = new Optimal_Leaf_Order(this._A, metric);
                result = this._optimal_leaf_order.ordering;
                break;

            case "spectral_order":
                if (!this._spectral_order) 
                    this._spectral_order = new Spectral_Order(this._A, metric);
                result = this._spectral_order.ordering;
                break;
            case "barycenter_order":
                if (!this._barycenter_order) 
                    this._barycenter_order = new Barycenter_Order(this._A, metric);
                result = this._barycenter_order.ordering;
                break;
        }
        return result;
    }
}

class Barycenter_Order{    
    constructor(A, metric = euclidean) {
        this._A = A;
        this._metric = metric;
        const N = this._N = A.shape[0];
        const [min, max] = this._get_extent(A);
        const sigma = Math.abs(max - min) / 2;
        // quad
        //const f = (v) => 1 - (v < 0 ? v / min : v / max) ** 2 //1 - (v / m) ** 2;
        // triangle
        //const f = (v) => 1 - (1 / Math.abs(v < 0 ? min : max) * v);
        // gauß
        const f = (v) => /* 1 / (Math.sqrt(2 * Math.PI * sigma)) * */ Math.exp(-(v ** 2 / (2 * sigma **2)));
        const bt = new BallTree(A.to2dArray);
        const B = new Array(N).fill(0);
        let ordering = this._ordering = linspace(0, N - 1);
        let max_iter = 100;
        while (--max_iter) {
            for (let i = 0; i < N; ++i) {
                const oi = ordering[i];
                const Ai = A.row(oi);
                let b = []; //median
                /* for (let j = 0; j < N; ++j) {
                    if (i == j) continue;
                    const oj = ordering[j];
                    const Aij = Ai[j];
                    if ((Aij > min / 10 && Aij < max / 10)) continue
                    //if (Aij < 2 * a_hat) {
                    //if (Aij > p_l && Aij < p_h) {
                        //b += 1/Aij;
                        //n++
                    //    b.push(j); //median
                    //}

                    // BETWEEN QUARTILES
                    //if (Aij > a_p25 && Aij < a_p75) {
                    //    b += Aij;
                    //    n++;
                    //}

                    // WEIGHTED BARYCENTER
                    //const w = f(Aij)
                    //b += w * oj;
                    //n += w;

                    // MIN
                    //if (Math.min(b, Aij) == Aij)
                    //    n = j;
                    //b = Math.min(b, Aij);
                } */
                //B[oi] = neumair_sum(b) / b.length; // mean
                const h = bt.search(Ai, 10);
                while (h.length > 1) b.push(h.pop());
                B[oi] = b.sort((a, b) => 
                    f(Ai[a.element.index]) - f(Ai[b.element.index])
                )[Math.floor(b.length / 2)].element.index; //median
                
                //B[oi] = b / n;
                
                //console.log(B[oi])
                //B[oi] = n; // MIN
            }
            //console.log(this._count_crossings([p_l, p_h]))
            console.log(ordering.map(d => B[d]));
            ordering = ordering.sort((a, b) => {
                const d = B[b] - B[a];
                if (d < 0) return 1;
                else if (d > 0) return -1;
                return 0;
            });
        }
        return this;
    }

    _get_percentiles(A, [p1, p2]) {
        const data = A.to2dArray.flat().sort((a, b) => a - b);
        const N = data.length;
        return [
            data[Math.floor(N * p1)],
            data[Math.floor(N * p2) - 1],
        ]
    }

    _get_extent(A) {
        const data = A.to2dArray.flat().sort((a, b) => a - b);
        const N = data.length;
        return [data[0], data[N - 1]];
    }

    _count_crossings([p1, p2]) {
        const A = this._A;
        const N = A.shape[0];
        const ordering = this._ordering;
        const south_sequence = [];

        for (let i = 0; i < N; ++i) {
            const oi = ordering[i];
            const Ai = A.row(oi);
            const sequence_i = [];
            for (let j = 0; j < N; ++j) {
                const oj = ordering[j];
                const Aij = Ai[oj];
                /*if (Aij < p1 && Aij > p2) {
                    sequence_i.push(oj);
                }*/
            }
            south_sequence.push(sequence_i.sort((a, b) => a - b));
        }

        let firstIndex = 1;
        while (firstIndex < N) firstIndex *= 2;
        const treeSize = 2 * firstIndex - 1;
        firstIndex -= 1;
        const tree = new Array(treeSize).fill(0);

        let crosscount = 0;
        for (let i = 0, n = south_sequence.length; i < n; ++i) {
            let index = south_sequence[i] + firstIndex;
            tree[index]++;
            while (index > 0) {
                if (index % 2) crosscount += tree[index + 1];
                index = (index - 1) / 2;
                tree[index]++;
            }
        }
        return crosscount;
    }

    get ordering() {
        return this._ordering;
    }
}

class Spectral_Order{
    constructor(A, metric=euclidean) {
        this._A = A;
        this._metric = metric;
        this._N = A.shape[0];
        const fiedler_vector = this._fiedler_vector(A);
        this._ordering = linspace(0, this._N - 1).sort((a, b) => fiedler_vector[a] - fiedler_vector[b]);
        return this;
    }

    get ordering() {
        return this._ordering;
    }

    _fiedler_vector(B) {
        const g = this._gershgorin_bound(B);
        const N = B.shape[0];
        const B_hat = new Matrix(N, N, (i, j) => i === j ? g - B.entry(i, j) : -B.entry(i, j));
        const eig = simultaneous_poweriteration(B_hat, 2);
        return eig.eigenvectors[1]
    }

    _gershgorin_bound(B) {
        let max = 0;
        let N = B.shape[0];
        for (let i = 0; i < N; ++i) {
            let t = B.entry(i, i);
            for (let j = 0; j < N; ++j) {
                if (i !== j) {
                    t += Math.abs(B.entry(i, j));
                }
            }
            max = max > t ? max : t;
        }
        return max;
    }
}

class Optimal_Leaf_Order{
    constructor(A, metric=euclidean) {
        this._A = A;
        const N = A.shape[0];
        const hclust = this._hclust = new Hierarchical_Clustering(A, "complete", metric);
        const distance_matrix = this._distance_matrix = new Array(N);
        for (let i = 0; i < N; ++i) {
            distance_matrix[i] = new Float64Array(N);
            for (let j = 0; j < N; ++j) {
                distance_matrix[i][j] = i === j ? Infinity : metric(A.row(i), A.row(j));
            }
        }
        this._order_map = new Map();
        let min = Infinity;
        this._optimal_order = null;
        let left = hclust.root.left.leaves();
        let right = hclust.root.right.leaves();

        for (let i = 0, n = left.length; i < n; ++i) {
            for (let j = 0, m = right.length; j < m; ++j) {
                let so = this.order(hclust.root, left[i], right[j]);
                if (so[0] < min) {
                    min = so[0];
                    this._optimal_order = so[1];
                }
            }
        }
        return this;
    }

    get ordering() {
        return this._optimal_order;
    }

    order(v, i, j) {
        const order_map = this._order_map;
        const key = `k${v.id}-${i}-${j}`; // ugly key
        /*if (key in order_map) 
            return order_map[key];
        return (order_map[key] = this._order(v, i, j))*/
        /*if (order_map.has(v)) {
            const v_map = order_map.get(v)
            if (v_map.has(`${i},${j}`)) {
                return v_map.get(`${i},${j}`)
            } else {
                let value = this._order(v, i, j);
                v_map.set(`${i},${j}`, value);
                return value;
            }
        } else {
            let value = this._order(v, i, j);
            const v_map = new Map();
            v_map.set(`${i},${j}`, value);
            order_map.set(v, v_map);
            return value;
        }*/
        if (order_map.has(key)) {
            return order_map.get(key);
        } else {
            let value = this._order(v, i, j);
            order_map.set(key, value);
            return value;
        }
    }

    _order(v, i, j) {
        if (v.isLeaf) {
            return [0, [v.index]];
        }
        const D = this._distance_matrix;
        let l = v.left;
        let r = v.right;
        let L = l ? l.leaves() : [];
        let R = r ? r.leaves() : [];
        let w;
        let x;
        if (L.indexOf(i) !== -1 && R.indexOf(j) !== -1) {
            w = l; 
            x = r;
        } else if (R.indexOf(i) !== -1 && L.indexOf(j) !== -1) {
            w = r;
            x = l;
        } else {
            throw "Node is not common ancestor of i and j";
        }

        let Wl = w.left ? w.left.leaves() : [];
        let Wr = w.right ? w.right.leaves() : [];
        let Ks = Wr.indexOf(i) != -1 ? Wl : Wr;
        if (Ks.length === 0) { 
            Ks = [i];
        }

        let Xl = x.left ? x.left.leaves() : [];
        let Xr = x.right ? x.right.leaves() : [];
        let Ls = Xr.indexOf(j) != -1 ? Xl : Xr;
        if (Ls.length === 0) {
            Ls = [j];
        }

        let min = Infinity;
        let optimal_order = [];
        for (let k = 0, Ks_length = Ks.length; k < Ks_length; ++k) {
            let w_min = this.order(w, i, Ks[k]);
            for (let m = 0, Ls_length = Ls.length; m < Ls_length; ++m) {
                let x_min = this.order(x, Ls[m], j);
                let dist = w_min[0] + D[Ks[k]][Ls[m]] + x_min[0];
                if (dist < min) {
                    min = dist;
                    optimal_order = w_min[1].concat(x_min[1]);
                }
            }
        }

        return [min, optimal_order];
    }
}

/**
 * @module matrix
 */

class Randomizer {
    // https://github.com/bmurray7/mersenne-twister-examples/blob/master/javascript-mersenne-twister.js
    /**
     * Mersenne Twister
     * @param {*} _seed 
     */
    constructor(_seed) {
        this._N = 624;
        this._M = 397;
        this._MATRIX_A = 0x9908b0df;
        this._UPPER_MASK = 0x80000000;
        this._LOWER_MASK = 0x7fffffff;
        this._mt = new Array(this._N);
        this._mti = this.N + 1;

        this.seed = _seed || new Date().getTime();
        return this;
    }

    set seed(_seed) {
        this._seed = _seed;
        let mt = this._mt;

        mt[0] = _seed >>> 0;
        for (this._mti = 1; this._mti < this._N; this._mti += 1) {
            let mti = this._mti;
            let s = mt[mti - 1] ^ (mt[mti - 1] >>> 30);
            mt[mti] = (((((s & 0xffff0000) >>> 16) * 1812433253) << 16) + (s & 0x0000ffff) * 1812433253) + mti;
            mt[mti] >>>= 0;
        }
    }

    get seed() {
        return this._seed;
    }

    get random() {
        return this.random_int * (1.0 / 4294967296.0)
    }

    get random_int() {
        let y, mag01 = new Array(0x0, this._MATRIX_A);
        if (this._mti >= this._N) {
            let kk;

            if (this._mti == this._N + 1) {
                this.seed = 5489;
            }

            let N_M = this._N - this._M;
            let M_N = this._M - this._N;

            for (kk = 0; kk < N_M; ++kk) {
                y = (this._mt[kk] & this._UPPER_MASK) | (this._mt[kk + 1] & this._LOWER_MASK);
                this._mt[kk] = this._mt[kk + this._M] ^ (y >>> 1) ^ mag01[y & 0x1];
            }
            for (; kk < this._N - 1; ++kk) {
                y = (this._mt[kk] & this._UPPER_MASK) | (this._mt[kk + 1] & this._LOWER_MASK);
                this._mt[kk] = this._mt[kk + M_N] ^ (y >>> 1) ^ mag01[y & 0x1];
            }

            y = (this._mt[this._N - 1] & this._UPPER_MASK) | (this._mt[0] & this._LOWER_MASK);
            this._mt[this._N - 1] = this._mt[this._M - 1] ^ (y >>> 1) ^ mag01[y & 0x1];

            this._mti = 0;
        }

        y = this._mt[this._mti += 1];
        y ^= (y >>> 11);
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= (y >>> 18);

        return y >>> 0;
    }

    choice(A, n) {
        if (A instanceof Matrix) {
            let [rows, cols] = A.shape;
            if (n > rows) throw "n bigger than A!";
            let sample = new Array(n);
            let index_list = linspace(0, rows - 1);
            for (let i = 0, l = index_list.length; i < n; ++i, --l) {
                let random_index = this.random_int % l;
                sample[i] = index_list.splice(random_index, 1)[0];
            }
            return sample.map(d => A.row(d));
        } else if (Array.isArray(A) || A instanceof Float64Array) {
            let rows = A.length;
            if (n > rows) {
                throw "n bigger than A!";
            }
            let sample = new Array(n);
            let index_list = linspace(0, rows - 1);
            for (let i = 0, l = index_list.length; i < n; ++i, --l) {
                let random_index = this.random_int % l;
                sample[i] = index_list.splice(random_index, 1)[0];
            }
            return sample.map(d => A[d]);
        }
    }

    static choice(A, n, seed=19870307) {
        let [rows, cols] = A.shape;
        if (n > rows) throw "n bigger than A!"
        let rand = new Randomizer(seed);
        let sample = new Array(n);
        let index_list = linspace(0, rows - 1);
        /*let index_list = new Array(rows);
        for (let i = 0; i < rows; ++i) {
            index_list[i] = i;
        }*/
        //let result = new Matrix(n, cols);
        for (let i = 0, l = index_list.length; i < n; ++i, --l) {
            let random_index = rand.random_int % l;
            sample[i] = index_list.splice(random_index, 1)[0];
            //random_index = index_list.splice(random_index, 1)[0];
            //result.set_row(i, A.row(random_index))
        }
        //return result;
        //return new Matrix(n, cols, (row, col) => A.entry(sample[row], col))
        return sample.map(d => A.row(d));
    }
}

class PCA{
    constructor(X, d=2) {
        this.X = X;
        this.d = d;
    }

    transform() {
        let X = this.X;
        let D = X.shape[1];
        let O = new Matrix(D, D, "center");
        let X_cent = X.dot(O);

        let C = X_cent.transpose().dot(X_cent);
        let { eigenvectors: V } = simultaneous_poweriteration(C, this.d);
        V = Matrix.from(V).transpose();
        this.Y = X.dot(V);
        return this.Y
    }

    get projection() {
        return this.Y
    }

    

}

class MDS{
    constructor(X, d=2, metric=euclidean) {
        this.X = X;
        this.d = d;
        this._metric = metric;
    }

    transform() {
        const X = this.X;
        //let sum_reduce = (a,b) => a + b
        const rows = X.shape[0];
        const metric = this._metric;
        let ai_ = [];
        let a_j = [];
        for (let i = 0; i < rows; ++i) {
            ai_.push(0);
            a_j.push(0);
        }
        let a__ = 0;
        const A = new Matrix();
        A.shape = [rows, rows, (i,j) => {
            let val = 0;
            if (i < j) {
                val = metric(X.row(i), X.row(j));
            } else if (i > j) {
                val = A.entry(j,i);
            }
            ai_[i] += val;
            a_j[j] += val;
            a__ += val;
            return val;
        }];
        this._d_X = A;
        ai_ = ai_.map(v => v / rows);
        a_j = a_j.map(v => v / rows);
        a__ /= (rows ** 2);
        const B = new Matrix(rows, rows, (i, j) => (A.entry(i, j) - ai_[i] - a_j[j] + a__));
        //B.shape = [rows, rows, (i,j) => (A.entry(i,j) - (A.row(i).reduce(sum_reduce) / rows) - (A.col(j).reduce(sum_reduce) / rows) + a__)]
                
        const { eigenvectors: V } = simultaneous_poweriteration(B, this.d);
        this.Y = Matrix.from(V).transpose();
        
        return this.Y
    }

    get projection() {
        return this.Y
    }

    get stress() {
        const N = this.X.shape[0];
        const Y = this.Y;
        const d_X = this._d_X; /*new Matrix();
        d_X.shape = [N, N, (i, j) => {
            return i < j ? metric(X.row(i), X.row(j)) : d_X.entry(j, i);
        }]*/
        const d_Y = new Matrix();
        d_Y.shape = [N, N, (i, j) => {
            return i < j ? euclidean(Y.row(i), Y.row(j)) : d_Y.entry(j, i);
        }];
        let top_sum = 0;
        let bottom_sum = 0;
        for (let i = 0; i < N; ++i) {
            for (let j = i + 1; j < N; ++j) {
                top_sum += Math.pow(d_X.entry(i, j) - d_Y.entry(i, j), 2);
                bottom_sum += Math.pow(d_X.entry(i, j), 2);
            }
        }
        return Math.sqrt(top_sum / bottom_sum);
    }
}

/**
 * @class
 * @alias ISOMAP
 */
class ISOMAP{

    /**
     * 
     * @constructor
     * @memberof module:dimensionality_reduction
     * @alias ISOMAP
     * @param {Matrix} X - the high-dimensional data. 
     * @param {number} neighbors - the number of neighbors {@link ISOMAP} should use to project the data.
     * @param {number} [d = 2] - the dimensionality of the projection. 
     * @param {function} [metric = euclidean] - the metric which defines the distance between two points. 
     */
    constructor(X, neighbors, d = 2, metric = euclidean) {
        this.X = X;
        this.k = neighbors || Math.floor(X.shape[0] / 10);
        this.d = d;
        this._metric = metric;
    }

    /**
     * Computes the projection.
     * @returns {Matrix} Returns the projection.
     */
    transform() {
        let X = this.X;
        let rows = X.shape[0];
        // make knn extern and parameter for constructor or transform?
        let D = new Matrix();
        D.shape = [rows, rows, (i,j) => i <= j ? this._metric(X.row(i), X.row(j)) : D.entry(j,i)];
        let kNearestNeighbors = [];
        for (let i = 0; i < rows; ++i) {
            let row = D.row(i).map((d,i) => { 
                return {
                    "index": i,
                    "distance": d
                }
            });
            let H = new Heap(row, d => d.distance, "min");
            kNearestNeighbors.push(H.toArray().slice(1, this.k + 1));
        }
        
        /*D = dijkstra(kNearestNeighbors);*/
        // compute shortest paths
        // TODO: make extern
        /** @see {@link https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm} */
        let G = new Matrix(rows, rows, (i,j) => {
            let other = kNearestNeighbors[i].find(n => n.index === j);
            return other ? other.distance : Infinity
        });

        for (let i = 0; i < rows; ++i) {
            for (let j = 0; j < rows; ++j) {
                for (let k = 0; k < rows; ++k) {
                    G.set_entry(i, j, Math.min(G.entry(i, j), G.entry(i, k) + G.entry(k, j)));
                }
            }
        }
        
        let ai_ = [];
        let a_j = [];
        for (let i = 0; i < rows; ++i) {
            ai_.push(0);
            a_j.push(0);
        }
        let a__ = 0;
        let A = new Matrix(rows, rows, (i,j) => {
            let val = G.entry(i, j);
            val = val === Infinity ? 0 : val;
            ai_[i] += val;
            a_j[j] += val;
            a__ += val;
            return val;
        });
        
        ai_ = ai_.map(v => v / rows);
        a_j = a_j.map(v => v / rows);
        a__ /= (rows ** 2);
        let B = new Matrix(rows, rows, (i,j) => (A.entry(i,j) - ai_[i] - a_j[j] + a__));
             
        // compute d eigenvectors
        let { eigenvectors: V } = simultaneous_poweriteration(B, this.d);
        this.Y = Matrix.from(V).transpose();
        // return embedding
        return this.Y
    }

    /**
     * @returns {Matrix} Returns the projection.
     */
    get projection() {
        return this.Y
    }

    

}

/**
 * @class
 * @alias FASTMAP
 */
class FASTMAP{
    /**
     * 
     * @constructor
     * @memberof module:dimensionality_reduction
     * @alias FASTMAP
     * @param {Matrix} X - the high-dimensional data. 
     * @param {number} [d = 2] - the dimensionality of the projection.
     * @param {function} [metric = euclidean] - the metric which defines the distance between two points.  
     * @returns {FASTMAP}
     */
    constructor(X, d=2, metric=euclidean) {
        this.X = X;
        this.d = d;
        this._metric = metric;
        this._col = -1;
        this.randomizer = new Randomizer(1212);
    }

    /**
     * Chooses two points which are the most distant in the actual projection.
     * @private
     * @param {function} dist 
     * @returns {Array} An array consisting of first index, second index, and distance between the two points.
     */
    _choose_distant_objects(dist) {
        let X = this.X;
        let N = X.shape[0];
        let a_index = this.randomizer.random_int % N - 1;
        let b_index = null;
        let max_dist = -Infinity;
        for (let i = 0; i < N; ++i) {
            let d_ai = dist(a_index, i);
            if (d_ai > max_dist) {
                max_dist = d_ai;
                b_index = i;
            }
        }
        max_dist = -Infinity;
        for (let i = 0; i < N; ++i) {
            let d_bi = dist(b_index, i);
            if (d_bi > max_dist) {
                max_dist = d_bi;
                a_index = i;
            }
        }
        return [a_index, b_index, max_dist];
    }

    /**
     * Computes the projection.
     * @returns {Matrix} The {@link d}-dimensional projection of the data matrix {@link X}.
     */
    transform() {
        let X = this.X;
        let [ rows, D ] = X.shape;
        let Y = new Matrix(rows, this.d);
        //let PA = [[], []];
        let dist = (a,b) => this._metric(X.row(a), X.row(b));
        let old_dist = dist;

        while(this._col < this.d - 1) {
            this._col += 1;
            let col = this._col;
            // choose pivot objects
            let [a_index, b_index, d_ab] = this._choose_distant_objects(dist);
            // record id of pivot objects
            //PA[0].push(a_index);
            //PA[1].push(b_index);
            if (d_ab === 0) {
                // because all inter-object distances are zeros
                for (let i = 0; i < rows; ++i) {
                    Y.set_entry(i, col, 0);
                }
            } else {
                // project the objects on the line (O_a, O_b)
                for (let i = 0; i < rows; ++i) {
                    let d_ai = dist(a_index, i);
                    let d_bi = dist(b_index, i);
                    let y_i = (d_ai ** 2 + d_ab ** 2 - d_bi ** 2) / (2 * d_ab);
                    Y.set_entry(i, col, y_i);
                }
                // consider the projections of the objects on a
                // hyperplane perpendicluar to the line (a, b);
                // the distance function D'() between two 
                // projections is given by Eq.4
                dist = (a,b) => Math.sqrt((old_dist(a,b) ** 2) - ((Y.entry(a, col) - Y.entry(b, col)) ** 2));
            }
        }
        // return embedding
        this.Y = Y;
        return this.Y;
    }

    /**
     * @returns {Matrix}
     */
    get projection() {
        return this.Y
    }

    

}

/**
 * @class
 * @alias LDA
 */
class LDA{

    /**
     * 
     * @constructor
     * @memberof module:dimensionality_reduction
     * @alias LDA
     * @param {Matrix} X - the high-dimensional data.
     * @param {Array} labels - the label / class of each data point.
     * @param {number} [d = 2] - the dimensionality of the projection.
     * @param {function} [metric = euclidean] - the metric which defines the distance between two points.  
     */
    constructor(X, labels, d = 2, metric = euclidean) {
        this.X = X;
        this._labels = labels;
        this.d = d;
        this._metric = metric;
    }

    transform() {
        let X = this.X;
        let [ rows, cols ] = X.shape;
        let labels = this._labels;
        let unique_labels = {};
        let label_id = 0;
        labels.forEach((l, i) => {
            if (l in unique_labels) {
                unique_labels[l].count++;
                unique_labels[l].rows.push(X.row(i));
            } else {
                unique_labels[l] = {
                    "id": label_id++,
                    "count": 1,
                    "rows": [X.row(i)]
                };
            }
        });
        
        // create X_mean and vector means;
        let X_mean = X.mean;
        let V_mean = new Matrix(label_id, cols);
        for (let label in unique_labels) {
            let V = Matrix.from(unique_labels[label].rows);
            let v_mean = V.meanCols;
            for (let j = 0; j < cols; ++j) {
                V_mean.set_entry(unique_labels[label].id, j, v_mean[j]);
            }           
        }
        // scatter_between
        let S_b = new Matrix(cols, cols);
        for (let label in unique_labels) {
            let v = V_mean.row(unique_labels[label].id);
            let m = new Matrix(cols, 1, (j) => v[j] - X_mean);
            let N = unique_labels[label].count;
            S_b = S_b.add(m.dot(m.transpose()).mult(N));
        }

        // scatter_within
        let S_w = new Matrix(cols, cols);
        for (let label in unique_labels) {
            let v = V_mean.row(unique_labels[label].id);
            let m = new Matrix(cols, 1, (j) => v[j]);
            let R = unique_labels[label].rows;
            for (let i = 0, n = unique_labels[label].count; i < n; ++i) {
                let row_v = new Matrix(cols, 1, (j,_) => R[i][j] - m.entry(j, 0));
                S_w = S_w.add(row_v.dot(row_v.transpose()));
            }
        }

        let { eigenvectors: V } = simultaneous_poweriteration(S_w.inverse().dot(S_b), this.d);
        V = Matrix.from(V).transpose();
        this.Y = X.dot(V);

        // return embedding
        return this.Y;
    }

    get projection() {
        return this.Y;
    }
}

class LLE{
    constructor(X, neighbors, d=2, metric=euclidean) {
        this.X = X;
        this._k = neighbors;
        this.d = d;
        this._metric = metric;
    }

    transform() {
        let X = this.X;
        let d = this.d;
        let [ rows, cols ] = X.shape;
        let k = this._k;
        let nN = k_nearest_neighbors(X.to2dArray, k, null, this._metric);
        let O = new Matrix(k, 1, 1);
        let W = new Matrix(rows, rows);

        for (let row = 0; row < rows; ++row) {
            let Z = new Matrix(k, cols, (i, j) => X.entry(nN[row][i].j, j) - X.entry(row, j));
            let C = Z.dot(Z.transpose());
            if ( k > cols ) {
                let C_trace = neumair_sum(C.diag) / 1000;
                for (let j = 0; j < k; ++j) {
                    C.set_entry(j, j, C.entry(j, j) + C_trace);
                }
            }

            // reconstruct;
            let w = Matrix.solve(C, O);
            let w_sum = neumair_sum(w.col(0));
            w = w.divide(w_sum);
            for (let j = 0; j < k; ++j) {
                W.set_entry(row, nN[row][j].j, w.entry(j, 0));
            }
        }
        // comp embedding
        let I = new Matrix(rows, rows, "identity");
        let IW = I.sub(W);
        let M = IW.transpose().dot(IW);
        let { eigenvectors: V } = simultaneous_poweriteration(M.transpose().inverse(), d + 1);
        
        this.Y = Matrix.from(V.slice(1, 1 + d)).transpose();

        // return embedding
        return this.Y;
    }

    get projection() {
        return this.Y;
    }
}

class MLLE{
    constructor(X, neighbors, d=2, metric=euclidean) {
        this.X = X;
        this._k = neighbors;
        this.d = d;
        this._metric = metric;
    }

    transform() {
        let X = this.X;
        let d = this.d;
        let [ rows, cols ] = X.shape;
        let k = this._k;
        // 1.1 Determine a neighborset
        let nN = k_nearest_neighbors(X.to2dArray, k, null, this._metric);
        let O = new Matrix(k, 1, 1);
        let W = new Matrix(rows, k);
        let Phi = new Matrix(rows, rows);

        let V = new Array(rows);
        let Lambda = new Array(rows);
        let P = new Array(rows);

        for (let row = 0; row < rows; ++row) {
            let I_i = nN[row].map(n => n.j);
            let x_i = Matrix.from(X.row(row), "row");
            let X_i = Matrix.from(I_i.map(n => X.row(n)));
            X_i = X_i.sub(x_i);
            //X_i = X_i.dot(new Matrix(X_i._cols, X_i._cols, "center"))
            let C_i = X_i.dot(X_i.transpose()); // k by k

            let gamma = neumair_sum(C_i.diag) / 1000;
            for (let j = 0; j < k; ++j) {
                C_i.set_entry(j, j, C_i.entry(j, j) + gamma);
            }
            
            let { eigenvalues: Lambdas, eigenvectors: v } = simultaneous_poweriteration(C_i, k);
            V[row] = v; // k by k, rows are eigenvectors, big to small
            Lambda[row] = Lambdas; // 1 by k, cols are eigenvalues, big to small
            P.push(neumair_sum(Lambdas.slice(d + 1)) / neumair_sum(Lambdas.slice(0, d)));

            // reconstruct;
            let w = Matrix.solve(C_i, O); // k by 1
            let w_sum = neumair_sum(w.col(0));
            w = w.divide(w_sum);
            for (let j = 0; j < k; ++j) {
                W.set_entry(row, j, w.entry(j, 0));
            }
        }
        // find regularized weights // median
        let theta = P.sort((rho_i, rho_j) => rho_i - rho_j)[Math.ceil(rows / 2)];
        
        for (let row = 0; row < rows; ++row) {
            let I_i = nN[row].map(n => n.j);
            let Lambdas = Lambda[row]; // 1 by k
            let s_i = Lambdas.map((Lambda, l) => {
                    return {
                        "l": l,
                        "ratio": neumair_sum(Lambdas.slice(k - l + 1)) / neumair_sum(Lambdas.slice(0, k - l)),
                        "Lambda": Lambda
                    }
                });
            //console.log(s_i)
            s_i = s_i
                .filter(s => s.ratio < theta && s.l <= k - d)
                .map(s => s.l).pop() || d;
            let V_i = V[row]; // k by k
            V_i = V_i.slice(k - s_i); // s_i by k
            let alpha_i = (1 / Math.sqrt(s_i)) * norm(V_i[0].map((_, j) => neumair_sum(V_i.map(r => r[j]))));
            V_i = Matrix.from(V_i); // s_i by k
            
            //https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/manifold/locally_linear.py#L703

            let h = new Matrix(s_i, 1, alpha_i);
            let ones = new Matrix(k, 1, 1);
            h = h.sub(V_i.dot(ones));
            let h_norm = norm(h.col(0));
            h = h_norm < 1e-12 ? h.mult(0) : h.divide(h_norm);
            V_i = V_i.T;
            ones = new Matrix(s_i, 1, 1);
            let w_i = Matrix.from(W.row(row), "col");
            
            /*let H_i = new Matrix(s_i, s_i, "identity");
            H_i = H_i.sub(h.mult(2).outer(h));
            let W_i = V_i.sub(V_i.dot(h).dot(h.T).mult(2)).add(w_i.mult(1 - alpha_i))
            */
            let W_i = V_i.sub(V_i.dot(h).dot(h.T).mult(2)).add(w_i.mult(1 - alpha_i).dot(ones.T));
            
            W_i = W_i.dot(W_i.T);
            for (let i = 0; i < k + 1; ++i) {
                for (let j = 0; j < s_i; ++j) {
                    Phi.set_entry(I_i[i], I_i[j], Phi.entry(I_i[i], I_i[j]) - (i === j ? 1 : 0 ) + W_i.entry(i, j));
                }
            }
        }
        //let { eigenvectors: Y } = simultaneous_poweriteration(Phi.inverse(), d + 1);
        //this.Y = Matrix.from(Y.slice(1)).transpose()

        let { eigenvectors: Y } = simultaneous_poweriteration(Phi, d + 1);
        this.Y = Matrix.from(Y.slice(1)).transpose();

        // return embedding
        return this.Y;
    }

    get projection() {
        return this.Y;
    }
}

// https://epubs.siam.org/doi/abs/10.1137/S1064827502419154
class LTSA{

    constructor(X, neighbors, d=2, metric=euclidean) {
        this.X = X;
        this._k = neighbors;
        this.d = d;
        this._metric = metric;
    }

    transform() {
        let X = this.X;
        let d = this.d;
        let [ rows, D ] = X.shape;
        let k = this._k;
        // 1.1 determine k nearest neighbors
        let nN = k_nearest_neighbors(X.to2dArray, k, null, this._metric);
        // center matrix
        let O = new Matrix(D, D, "center");
        let B = new Matrix(rows, rows, 0);
        
        for (let row = 0; row < rows; ++row) {
            // 1.2 compute the d largest eigenvectors of the correlation matrix
            let I_i = [row, ...nN[row].map(n => n.j)];
            let X_i = Matrix.from(I_i.map(n => X.row(n)));
            // center X_i
            X_i = X_i.dot(O);
            // correlation matrix
            let C = X_i.dot(X_i.transpose());
            let { eigenvectors: g } = simultaneous_poweriteration(C, d);
            //g.push(linspace(0, k).map(_ => 1 / Math.sqrt(k + 1)));
            let G_i_t = Matrix.from(g);
            // 2. Constructing alignment matrix
            let W_i = G_i_t.transpose().dot(G_i_t).add(1 / Math.sqrt(k + 1));
            for (let i = 0; i < k + 1; ++i) {
                for (let j = 0; j < k + 1; ++j) {
                    B.set_entry(I_i[i], I_i[j], B.entry(I_i[i], I_i[j]) - (i === j ? 1 : 0 ) + W_i.entry(i, j));
                }
            }
        }

        // 3. Aligning global coordinates
        let { eigenvectors: Y } = simultaneous_poweriteration(B, d + 1);
        this.Y = Matrix.from(Y.slice(1)).transpose();

        // return embedding
        return this.Y;
    }

    get projection() {
        return this.Y;
    }
}

/*import { simultaneous_poweriteration} from "../linear_algebra/index";
import { k_nearest_neighbors } from "../matrix/index";
import { neumair_sum } from "../numerical/index";
import { norm } from "../matrix/index";
import { linspace } from "../matrix/index";*/

class TSNE{
    constructor(X, perplexity, epsilon, d=2, metric=euclidean, seed=1212) {
        this._X = X;
        this._d = d;
        [ this._N, this._D ] = X.shape;
        this._perplexity = perplexity;
        this._epsilon = epsilon;
        this._metric = metric;
        this._iter = 0;
        this.randomizer = new Randomizer(seed);
        this._Y = new Matrix(this._N, this._d, () => this.randomizer.random);
    }

    init(distance_matrix=null) {
        // init
        let Htarget = Math.log(this._perplexity);
        let D = distance_matrix || new Matrix(this._N, this._N, (i, j) => this._metric(this._X.row(i), this._X.row(j)));
        let P = new Matrix(this._N, this._N, "zeros");

        this._ystep = new Matrix(this._N, this._D, "zeros").to2dArray;
        this._gains = new Matrix(this._N, this._D, 1).to2dArray;

        // search for fitting sigma
        let prow = new Array(this._N).fill(0);
        for (let i = 0, N = this._N; i < N; ++i) {
            let betamin = -Infinity;
            let betamax = Infinity;
            let beta = 1;
            let done = false;
            let maxtries = 50;
            let tol = 1e-4;

            let num = 0;
            while(!done) {
                let psum = 0;
                for (let j = 0; j < N; ++j) {
                    let pj = Math.exp(-D.entry(i, j) * beta);
                    if (i === j) pj = 0;
                    prow[j] = pj;
                    psum += pj;
                }
                let Hhere = 0;
                for (let j = 0; j < N; ++j) {
                    let pj = (psum === 0) ? 0 : prow[j] / psum;
                    prow[j] = pj;
                    if (pj > 1e-7) Hhere -= pj * Math.log(pj);
                }
                if (Hhere > Htarget) {
                    betamin = beta;
                    beta = (betamax === Infinity) ? (beta * 2) : ((beta + betamax) / 2);
                } else {
                    betamax = beta;
                    beta = (betamin === -Infinity) ? (beta / 2) : ((beta + betamin) / 2);
                }
                ++num;
                if (Math.abs(Hhere - Htarget) < tol) done = true;
                if (num >= maxtries) done = true;
            }

            for (let j = 0; j < N; ++j) {
                P.set_entry(i, j, prow[j]);
            }
        }

        //compute probabilities
        let Pout = new Matrix(this._N, this._N, "zeros");
        let N2 = this._N * 2;
        for (let i = 0, N = this._N; i < N; ++i) {
            for (let j = 0; j < N; ++j) {
                Pout.set_entry(i, j, Math.max((P.entry(i, j) + P.entry(j, i)) / N2, 1e-100));
            }
        }
        this._P = Pout;
        return this
    }

    set perplexity(value) {
        this._perplexity = value;
    }

    get perplexity() {
        return this._perplexity;
    }

    set epsilon(value) {
        this._epsilon = value;
    }

    get epsilon() {
        return this._epsilon;
    }

    transform(iterations=1000) {
        for (let i = 0; i < iterations; ++i) {
            this.next();
        }
        return this._Y;
    }

    * transform_iter() {
        while (true) {
            this.next();
            yield {
                "projection": this._Y,
                "iteration": this._iter,
            };
        }
    }

    // perform optimization
    next() {
        let iter = ++this._iter;
        let P = this._P;
        let ystep = this._ystep;
        let gains = this._gains;
        let Y = this._Y;
        let N = this._N;
        let epsilon = this._epsilon;
        let dim = this._d;

        //calc cost gradient;
        let pmul = iter < 100 ? 4 : 1;
        
        // compute Q dist (unnormalized)
        let Qu = new Matrix(N, N, "zeros");
        let qsum = 0;
        for (let i = 0; i < N; ++i) {
            for (let j = i + 1; j < N; ++j) {
                let dsum = 0;
                for (let d = 0; d < dim; ++d) {
                    let dhere = Y.entry(i, d) - Y.entry(j, d);
                    dsum += dhere * dhere;
                }
                let qu = 1 / (1 + dsum);
                Qu.set_entry(i, j, qu);
                Qu.set_entry(j, i, qu);
                qsum += 2 * qu;
            }
        }

        // normalize Q dist
        let Q = new Matrix(N, N, (i, j) => Math.max(Qu.entry(i, j) / qsum, 1e-100));

        let cost = 0;
        let grad = [];
        for (let i = 0; i < N; ++i) {
            let gsum = new Array(dim).fill(0);
            for (let j = 0; j < N; ++j) {
                cost += -P.entry(i, j) * Math.log(Q.entry(i, j));
                let premult = 4 * (pmul * P.entry(i, j) - Q.entry(i, j)) * Qu.entry(i, j);
                for (let d = 0; d < dim; ++d) {
                    gsum[d] += premult * (Y.entry(i, d) - Y.entry(j, d));
                }
            }
            grad.push(gsum);
        }

        // perform gradient step
        let ymean = new Array(dim).fill(0);
        for (let i = 0; i < N; ++i) {
            for (let d = 0; d < dim; ++d) {
                let gid = grad[i][d];
                let sid = ystep[i][d];
                let gainid = gains[i][d];
                
                let newgain = Math.sign(gid) === Math.sign(sid) ? gainid * .8 : gainid + .2;
                if (newgain < .01) newgain = .01;
                gains[i][d] = newgain;

                let momval = iter < 250 ? .5 : .8;
                let newsid = momval * sid - epsilon * newgain * grad[i][d];
                ystep[i][d] = newsid;

                Y.set_entry(i, d, Y.entry(i, d) + newsid);
                ymean[d] += Y.entry(i, d);
            }
        }

        for (let i = 0; i < N; ++i) {
            for (let d = 0; d < 2; ++d) {
                Y.set_entry(i, d, Y.entry(i, d) - ymean[d] / N);
            }
        }

        return this._Y;
    }

    get projection() {
        return this._Y;
    }
}

// http://optimization-js.github.io/optimization-js/optimization.js.html#line438
function powell(f, x0, max_iter=300) {
    const epsilon = 1e-2;
    const n = x0.length;
    let alpha = 1e-3;
    let pfx = 10000;
    let x = x0.slice();
    let fx = f(x);
    let convergence = false;
    
    while (max_iter-- >= 0 && !convergence) {
        convergence = true;
        for (let i = 0; i < n; ++i) {
            x[i] += 1e-6;
            let fxi = f(x);
            x[i] -= 1e-6;
            let dx = (fxi - fx) / 1e-6;
            if (Math.abs(dx) > epsilon) {
                convergence = false;
            }
            x[i] -= alpha * dx;
            fx = f(x);
        }
        alpha *= (pfx >= fx ? 1.05 : 0.4);
        pfx = fx;
    }
    return x;
}

class UMAP{
    constructor(X, local_connectivity, min_dist, d=2, metric=euclidean, seed=1212) {
        this._X = X;
        this._d = d;
        [ this._N, this._D ] = X.shape;
        this._local_connectivity = local_connectivity;
        this._min_dist = min_dist;
        this._metric = metric;
        this._iter = 0;
        this._n_neighbors = 11;
        this._spread = 1;
        this._set_op_mix_ratio = 1;
        this._repulsion_strength = 1;
        this._negative_sample_rate = 5;
        this._n_epochs = 350;
        this._initial_alpha = 1;
        this._randomizer = new Randomizer(seed);
        this._Y = new Matrix(this._N, this._d, () => this._randomizer.random);
    }

    _find_ab_params(spread, min_dist) {
        function curve(x, a, b) {
            return 1 / (1 + a * Math.pow(x, 2 * b));
        }
      
        var xv = linspace(0, spread * 3, 300);
        var yv = linspace(0, spread * 3, 300);
        
        for ( var i = 0, n = xv.length; i < n; ++i ) {
            if (xv[i] < min_dist) {
                yv[i] = 1;
            } else {
                yv[i] = Math.exp(-(xv[i] - min_dist) / spread);
            }
        }
      
        function err(p) {
            var error = linspace(1, 300).map((_, i) => yv[i] - curve(xv[i], p[0], p[1]));
            return Math.sqrt(neumair_sum(error.map(e => e * e)));
        }
      
        var [ a, b ] = powell(err, [1,1]);
        return [ a, b ]
    }

    _compute_membership_strengths(distances, sigmas, rhos) {
        for (let i = 0, n = distances.length; i < n; ++i) {
            for (let j = 0, m = distances[i].length; j < m; ++j) {
                let v = distances[i][j].value - rhos[i];
                let value = 1;
                if (v > 0) {
                    value = Math.exp(-v / sigmas[i]);
                }
                distances[i][j].value = value;
            }
        }
        return distances;
    }

    _smooth_knn_dist(knn, k) {
        const SMOOTH_K_TOLERANCE = 1e-5;
        const MIN_K_DIST_SCALE = 1e-3;
        const n_iter = 64;
        const local_connectivity = this._local_connectivity;
        const bandwidth = 1;
        const target = Math.log2(k) * bandwidth;
        const rhos = [];
        const sigmas = [];
        const X = this._X;

        let distances = [];
        for (let i = 0, n = X.shape[0]; i < n; ++i) {
            let x_i = X.row(i);
            distances.push(knn.search(x_i, Math.max(local_connectivity, k)).raw_data().reverse());
        }

        for (let i = 0, n = X.shape[0]; i < n; ++i) {
            let search_result = distances[i];
            rhos.push(search_result[0].value);

            let lo = 0;
            let hi = Infinity;
            let mid = 1;

            for (let x = 0; x < n_iter; ++x) {
                let psum = 0;
                for (let j = 0; j < k; ++j) {
                    let d = search_result[j].value - rhos[i];
                    psum += (d > 0 ? Math.exp(-(d / mid)) : 1);
                }
                if (Math.abs(psum - target) < SMOOTH_K_TOLERANCE) {
                    break;
                }
                if (psum > target) {
                    //[hi, mid] = [mid, (lo + hi) / 2];
                    hi = mid;
                    mid = (lo + hi) / 2; // PROBLEM mit hi?????
                } else {
                    lo = mid;
                    if (hi === Infinity) {
                        mid *= 2;
                    } else {
                        mid = (lo + hi) / 2;
                    }
                }
            }
            sigmas[i] = mid;

            const mean_ithd = search_result.reduce((a, b) => a + b.value, 0) / search_result.length;
            //let mean_d = null;
            if (rhos[i] > 0) {
                if (sigmas[i] < MIN_K_DIST_SCALE * mean_ithd) {
                    sigmas[i] = MIN_K_DIST_SCALE * mean_ithd;
                }
            } else {
                const mean_d = distances.reduce((acc, res) => acc + res.reduce((a, b) => a + b.value, 0) / res.length);
                if (sigmas[i] > MIN_K_DIST_SCALE * mean_d) {
                    sigmas[i] = MIN_K_DIST_SCALE * mean_d;
                }
                
            }
        }
        return {distances: distances, sigmas: sigmas, rhos: rhos}
    }

    _fuzzy_simplicial_set(X, n_neighbors) {
        const knn = new BallTree(X.to2dArray, euclidean);
        let { distances, sigmas, rhos } = this._smooth_knn_dist(knn, n_neighbors);
        distances = this._compute_membership_strengths(distances, sigmas, rhos);
        let result = new Matrix(X.shape[0], X.shape[0], "zeros");
        for (let i = 0, n = X.shape[0]; i < n; ++i) {
            for (let j = 0; j < n_neighbors; ++j) {
                result.set_entry(i, distances[i][j].element.index, distances[i][j].value);
            }
        }
        const transposed_result = result.T;
        const prod_matrix = result.mult(transposed_result);
        result = result
            .add(transposed_result)
            .sub(prod_matrix)
            .mult(this._set_op_mix_ratio)
            .add(prod_matrix.mult(1 - this._set_op_mix_ratio));
        return result;
    }

    _make_epochs_per_sample(graph, n_epochs) {
        const { data: weights } = this._tocoo(graph);
        let result = new Array(weights.length).fill(-1);
        const weights_max = Math.max(...weights);
        const n_samples = weights.map(w => n_epochs * (w / weights_max));
        result = result.map((d, i) => (n_samples[i] > 0) ? Math.round(n_epochs / n_samples[i]) : d);
        return result;
    }

    _tocoo(graph) {
        const rows = [];
        const cols = [];
        const data = [];
        const [ rows_n, cols_n ] = graph.shape;
        for (let row = 0; row < rows_n; ++row) {
            for (let col = 0; col < cols_n; ++col) {
                const entry = graph.entry(row, col);
                if (entry !== 0) {
                    rows.push(row);
                    cols.push(col);
                    data.push(entry);
                }
            }
        }
        return {rows: rows, cols: cols, data: data};
    }

    init() {
        const [ a, b ] = this._find_ab_params(this._spread, this._min_dist);
        this._a = a;
        this._b = b;
        this._graph = this._fuzzy_simplicial_set(this._X, this._n_neighbors);
        this._epochs_per_sample = this._make_epochs_per_sample(this._graph, this._n_epochs);
        this._epochs_per_negative_sample = this._epochs_per_sample.map(d => d * this._negative_sample_rate);
        this._epoch_of_next_sample = this._epochs_per_sample.slice();
        this._epoch_of_next_negative_sample = this._epochs_per_negative_sample.slice();
        const { rows, cols } = this._tocoo(this._graph);
        this._head = rows;
        this._tail = cols;
        return this
    }

    set local_connectivity(value) {
        this._local_connectivity = value;
    }

    get local_connectivity() {
        return this._local_connectivity;
    }

    set min_dist(value) {
        this._min_dist = value;
    }

    get min_dist() {
        return this._min_dist;
    }

    transform(iterations = 1000) {
        for (let i = 0; i < iterations; ++i) {
            this.next();
        }
        return this._Y;
    }

    * transform_iter() {
        this._iter = 0;
        while (this._iter < this._n_epochs) {
            this.next();
            yield this._Y;
        }
        return this._Y;
    }

    _clip(x) {
        if (x > 4) return 4;
        if (x < -4) return -4;
        return x;
    }

    _optimize_layout(head_embedding, tail_embedding, head, tail) {
        const { 
            _d: dim, 
            _alpha: alpha, 
            _repulsion_strength: repulsion_strength, 
            _a: a, 
            _b: b,
            _epochs_per_sample: epochs_per_sample,
            _epochs_per_negative_sample: epochs_per_negative_sample,
            _epoch_of_next_negative_sample: epoch_of_next_negative_sample,
            _epoch_of_next_sample: epoch_of_next_sample,
            _clip: clip
        } = this;
        const tail_length = tail.length;

        for (let i = 0, n = epochs_per_sample.length; i < n; ++i) {
            if (epoch_of_next_sample[i] <= this._iter) {
                const j = head[i];
                const k = tail[i];
                const current = head_embedding.row(j);
                const other = tail_embedding.row(k);
                const dist = euclidean(current, other);//this._metric(current, other);
                let grad_coeff = 0;
                if (dist > 0) {
                    grad_coeff = (-2 * a * b * Math.pow(dist, b - 1)) / (a * Math.pow(dist, b) + 1);
                }
                for (let d = 0; d < dim; ++d) {
                    const grad_d = clip(grad_coeff * (current[d] - other[d])) * alpha;
                    const c = current[d] + grad_d;
                    const o = other[d] - grad_d;
                    current[d] = c;
                    other[d] = o;
                    head_embedding.set_entry(j, d, c);
                    tail_embedding.set_entry(k, d, o);
                }
                epoch_of_next_sample[i] += epochs_per_sample[i];
                const n_neg_samples = (this._iter - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i];
                for (let p = 0; p < n_neg_samples; ++p) {
                    const k = Math.floor(this._randomizer.random * tail_length);
                    const other = tail_embedding.row(tail[k]);
                    const dist = euclidean(current, other);//this._metric(current, other);
                    let grad_coeff = 0;
                    if (dist > 0) {
                        grad_coeff = (2 * repulsion_strength * b) / ((.01 + dist) * (a * Math.pow(dist, b) + 1));
                    } else if (j == k) {
                        continue;
                    }
                    for (let d = 0; d < dim; ++d) {
                        const grad_d = clip(grad_coeff * (current[d] - other[d])) * alpha;
                        const c = current[d] + grad_d;
                        const o = other[d] - grad_d;
                        current[d] = c;
                        other[d] = o;
                        head_embedding.set_entry(j, d, c);
                        tail_embedding.set_entry(tail[k], d, o);
                    }
                }
                epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i]);
            }
        }
        return head_embedding;
    }

    next() {
        let iter = ++this._iter;
        let Y = this._Y;

        this._alpha = (this._initial_alpha * (1 - iter / this._n_epochs));
        this._Y = this._optimize_layout(Y, Y, this._head, this._tail);

        return this._Y;
    }

    get projection() {
        return this._Y;
    }
}

//import { Matrix } from "../matrix/index";

/**
 * 
 */
class OAP {
    constructor(X, depth_field_lag, step_size, depth_weight, d = 2, metric = euclidean, seed = 1212) {
        this._X = X;
        this._d = d;
        [this._N, this._D] = X.shape;
        this._depth_field_lag = depth_field_lag;
        this._step_size = step_size;
        this._depth_weight = depth_weight;
        this._J = 3;
        this._max_iter = 1;
        this._metric = metric;
        this._seed = seed;
        this._randomizer = new Randomizer(seed);
    }

    _data_depth(technique = "chebyshev") {
        const X = this._X;
        const N = this._N;
        const h = new Float32Array(N);
        let deepest_point = 0;
        if (technique === "mdb") {
            h.fill(1);

            /*
            // Modified Band Depth 
            // https://www.tandfonline.com/doi/pdf/10.1198/jasa.2009.0108?casa_token=E1Uzntgs-5AAAAAA:Eo8mUpJDhpLQ5RHBkCB3Mdz0tbGM3Q0v78bwyCIAv7-peLGwfG3TcXLqShIaYuJLEqKc7GvaKlgvUg 
            const randomizer = this._randomizer;
            const h = new Float32Array(this._N);
            const J = this._J;
            const N = this._N;
            const D = this._D;
            const X = this._X;

            const one_div_by_n_choose_j = 1;
            for (let row = 0; row < N; ++row) {
                const x = X.row(row);
                const B_min = new Float32Array(D).fill(Infinity);
                const B_max = new Float32Array(D).fill(-Infinity);
                let r = Math.floor(randomizer.random * N);
                for (let i = 0; i < J; ++i) {
                    const x_j = X.row(r);
                    for (let d = 0; d < D; ++d) {
                        const x_jd = x_j[d]
                        B_min[d] = Math.min(B_min[d], x_jd);
                        B_max[d] = Math.max(B_max[d], x_jd);
                    }
                    r += Math.floor(randomizer.random * (N - 1));
                    r = r % N;
                }
                for (let d = 0; d < D; ++d) {
                    const x_d = x[d];
                    if (x_d >= B_min[d] && x_d <= B_max[d]) {
                        ++h[row]
                    }
                }
            }
            this._h = h;*/
        } else if (technique === "chebyshev") {
            // L∞ Depth
            // https://arxiv.org/pdf/1506.01332.pdf
            for (let i = 0; i < N; ++i) {
                let x = X.row(i);
                let sum = 0;
                for (let j = 0; j < N; ++j) {
                    if (i !== j) {
                        sum += chebyshev(x, X.row(j));
                    }
                }
                h[i] = 1 / (1 + sum / N);
                if (h[deepest_point] < h[i]) {
                    deepest_point = i;
                }
            }
        }
        this._h = h;
        this._deepest_point = deepest_point;

    }

    init() {
        this._iter = 0;
        // init with MDS
        const init_MDS = new MDS(this._X, this._d, this._metric);
        //console.log(init_MDS)
        this._Y = init_MDS.transform();

        // try häääh?
        this._X_distances = init_MDS._d_X;
        /*let max = -Infinity
        init_MDS._d_X._data.forEach(dx => max = Math.max(dx, max));
        this._X_distances = init_MDS._d_X.divide(max);*/
        // end try hääääh?
        
        // compute order statistics
        this._data_depth();
        this._M = this._monotonic_field(this._Y);
        //
        return this;
    }

    set depth_field_lag(value) {
        this._depth_field_lag = value;
    }

    get depth_field_lag() {
        return this._depth_field_lag;
    }

    set step_size(value) {
        this._step_size = value;
    }

    get step_size() {
        return this._step_size;
    }

    set depth_weight(value) {
        this._depth_weight = value;
    }

    get depth_weight() {
        return this._depth_weight;
    }

    transform(iterations = this._max_iter) {
        for (let i = 0; i < iterations; ++i) {
            this.next();
        }
        return this._Y;
    }

    * transform_iter() {
        while (true) {
            this.next();
            yield this._Y;
        }
    }

    _monotonic_field(Y) {
        const h = this._h;
        const Y_ = this._Y_;
        const nn = new BallTree();
        nn.add(Y.to2dArray);

        const N = 5;
        let M = (x) => {
            let neighbors = nn.search(x, N).toArray();
            let d_sum = 0;//neighbors.reduce((a, b) => a + b.value, 0);
            let m = 0;
            for (let i = 0; i < N; ++i) {
                d_sum += neighbors[i].value;
                m += h[neighbors[i].element.index] * neighbors[i].value;
            }
            //console.log(m, d_sum)
            m /= d_sum;
            return m;
        };
        return M;
    }

    next() {
        const iter = ++this._iter;
        const l = this._depth_field_lag;
        const step_size = this._step_size;
        const w = this._depth_weight;
        const N = this._N;
        const dim = this._d;
        const d_X = this._X_distances;
        const h = this._h;
        let Y = this._Y;

        if ((iter % l) === 1) {
            // compute monotonic field
            this._Y_ = this._Y.clone();
            this._M = this._monotonic_field(Y);
        }
        const M = this._M;
        // perform gradient step

        // MDS stress step
        /*for (let i = 0; i < N; ++i) {
            const d_x = d_X.row(i);
            const y_i = Y.row(i)
            const delta_mds_stress = new Float32Array(dim);
            for (let j = 0; j < N; ++j) {
                if (i !== j) {
                    const y_j = Y.row(j)
                    const d_y = metric(y_i, y_j);
                    const d_x_j = d_x[j] === 0 ? 1e-2 : d_x[j]
                    const mult = 1 - (d_x_j / d_y)
                    for (let d = 0; d < dim; ++d) {
                        delta_mds_stress[d] += (mult * (y_i[d] - y_j[d]));
                    }
                }
            }
            for (let d = 0; d < dim; ++d) {
                Y.set_entry(i, d, Y.entry(i, d) - step_size * delta_mds_stress[d] / N)
            }
        }*/
        
        // MDS stress step
        const d_Y = new Matrix();
        d_Y.shape = [N, N, (i, j) => {
            return i < j ? euclidean(Y.row(i), Y.row(j)) : d_Y.entry(j, i);
        }];
        const ratio = new Matrix();//d_X.divide(d_Y).mult(-1);
        ratio.shape = [N, N, (i, j) => {
            if (i === j) return 1e-8
            return i < j ? -d_X.entry(i, j) / d_Y.entry(i, j) : ratio.entry(j, i);
        }];
        for (let i = 0; i < N; ++i) {
            ratio.set_entry(i, i, ratio.entry(i, i) - neumair_sum(ratio.row(i)));
        }
        const mds_Y = ratio.dot(Y).divide(N);

        // Data depth step
        const diff_Y = new Matrix(N, dim, (i, j) => mds_Y.entry(i, j) - Y.entry(i, j));

        for (let i = 0; i < N; ++i) {
            const m = M(Y.row(i));
            const dm = M(mds_Y.row(i));
            const h_i = h[i];
            for (let d = 0; d < dim; ++d) {
                Y.set_entry(i, d, Y.entry(i, d) + step_size * (diff_Y.entry(i, d) + w * 2 * (m - h_i) * dm));
            }
        }

        this._Y = Y;

        return this._Y;
    }

    get projection() {
        return this._Y;
    }
}

/**
 * @class
 * @alias TriMap
 */
class TriMap{
    /**
     * 
     * @constructor
     * @memberof module:dimensionality_reduction
     * @alias TriMap
     * @param {Matrix} X - the high-dimensional data. 
     * @param {Number} [weight_adj = 500] - scaling factor.
     * @param {Number} [c = 5] - number of triplets multiplier.
     * @param {Number} [d = 2] - the dimensionality of the projection.
     * @param {Function} [metric = euclidean] - the metric which defines the distance between two points.  
     * @returns {TriMap}
     * @see {@link https://arxiv.org/pdf/1910.00204v1.pdf}
     * @see {@link https://github.com/eamid/trimap}
     */
    constructor(X, weight_adj = 500, c = 5, d = 2, metric = euclidean, randomizer = null) {
        this.X = X;
        this.d = d;
        this._metric = metric;
        this.randomizer = randomizer || new Randomizer(1212);
        this.weight_adj = weight_adj;
        this.c = c;
        return this;
    }

    /**
     * 
     * @param {Matrix} [pca = null] - Initial Embedding (if null then PCA gets used). 
     * @param {KNN} [knn = null] - KNN Object (if null then BallTree gets used). 
     */
    init(pca = null, knn = null) {
        const X = this.X;
        const N = X.shape[0];
        const d = this.d;
        const metric = this._metric;
        const c = this.c;
        this.n_inliers = 2 * c;
        this.n_outliers = 1 * c;
        this.n_random = 1 * c;
        this.Y = pca || new PCA(X, d).transform();//.mult(.01);
        this.knn = knn || new BallTree(X.to2dArray, metric);
        const {triplets, weights} = this._generate_triplets(this.n_inliers, this.n_outliers, this.n_random);
        this.triplets = triplets;
        this.weights = weights;
        this.lr = 1000 * N / triplets.shape[0];
        this.C = Infinity;
        this.tol = 1e-7;
        this.vel = new Matrix(N, d, 0);
        this.gain = new Matrix(N, d, 1);
        return this;
    }

    /**
     * Generates {@link n_inliers} x {@link n_outliers} x {@link n_random} triplets.
     * @param {Number} n_inliers 
     * @param {Number} n_outliers 
     * @param {Number} n_random 
     */
    _generate_triplets(n_inliers, n_outliers, n_random) {
        const metric = this._metric;
        const weight_adj = this.weight_adj;
        const X = this.X;
        const N = X.shape[0];
        const knn = this.knn;
        const n_extra = Math.min(n_inliers + 20, N);
        const nbrs = new Matrix(N, n_extra);
        const knn_distances = new Matrix(N, n_extra);
        for (let i = 0; i < N; ++i) {
            knn.search(X.row(i), n_extra + 1)
                .raw_data()
                .filter(d => d.value != 0)
                .sort((a, b) => a.value - b.value)
                .forEach((d, j) => {
                    nbrs.set_entry(i, j, d.element.index);
                    knn_distances.set_entry(i, j, d.value);
                });
        }
        // scale parameter
        const sig = new Float64Array(N);
        for (let i = 0; i < N; ++i) {
            sig[i] = Math.max(
                   (knn_distances.entry(i, 3) +
                    knn_distances.entry(i, 4) +
                    knn_distances.entry(i, 5) +
                    knn_distances.entry(i, 6)) / 4,
                    1e-10);
        }
        
        const P = this._find_p(knn_distances, sig, nbrs);
        
        let triplets = this._sample_knn_triplets(P, nbrs, n_inliers, n_outliers);
        let n_triplets = triplets.shape[0];
        const outlier_distances = new Float64Array(n_triplets);
        for (let i = 0; i < n_triplets; ++i) {
            const j = triplets.entry(i, 0);
            const k = triplets.entry(i, 2);
            outlier_distances[i] = metric(X.row(j), X.row(k));
        }
        let weights = this._find_weights(triplets, P, nbrs, outlier_distances, sig);
        
        if (n_random > 0) {
            const {random_triplets, random_weights} = this._sample_random_triplets(X, n_random, sig);
            triplets = triplets.concat(random_triplets, "vertical");
            weights = Float64Array.from([...weights, ...random_weights]);
        }
        n_triplets = triplets.shape[0];
        let max_weight = -Infinity;
        for (let i = 0; i < n_triplets; ++i) {
            if (isNaN(weights[i])) {weights[i] = 0;}
            if (max_weight < weights[i]) max_weight = weights[i];
        }
        let max_weight_2 = -Infinity;
        for (let i = 0; i < n_triplets; ++i) {
            weights[i] /= max_weight;
            weights[i] += .0001;
            weights[i] = Math.log(1 + weight_adj * weights[i]);
            if (max_weight_2 < weights[i]) max_weight_2 = weights[i];
        }
        for (let i = 0; i < n_triplets; ++i) {
            weights[i] /= max_weight_2;
        }
        return {
            "triplets": triplets,
            "weights": weights,
        }
    }

    /**
     * Calculates the similarity matrix P
     * @param {Matrix} knn_distances - matrix of pairwise knn distances
     * @param {Float64Array} sig - scaling factor for the distances
     * @param {Matrix} nbrs - nearest neighbors
     * @returns {Matrix} pairwise similarity matrix
     */
    _find_p(knn_distances, sig, nbrs) {
        const [N, n_neighbors] = knn_distances.shape;
        return new Matrix(N, n_neighbors, (i, j) => {
            return Math.exp(-((knn_distances.entry(i, j) ** 2) / sig[i] / sig[nbrs.entry(i, j)]));
        });
    }

    /**
     * Sample nearest neighbors triplets based on the similarity values given in P.
     * @param {Matrix} P - Matrix of pairwise similarities between each point and its neighbors given in matrix nbrs.
     * @param {Matrix} nbrs - Nearest neighbors indices for each point. The similarity values are given in matrix {@link P}. Row i corresponds to the i-th point.
     * @param {Number} n_inliers - Number of inlier points.
     * @param {Number} n_outliers - Number of outlier points.
     * 
     */
    _sample_knn_triplets(P, nbrs, n_inliers, n_outliers) {
        const N = nbrs.shape[0];
        const triplets = new Matrix(N * n_inliers * n_outliers, 3);
        for (let i = 0; i < N; ++i) {
            let n_i = i * n_inliers * n_outliers;
            const sort_indices = this.__argsort(P.row(i).map(d => -d));
            for (let j = 0; j < n_inliers; ++j) {
                let n_j = j * n_outliers;
                const sim = nbrs.entry(i, sort_indices[j]);
                const samples = this._rejection_sample(n_outliers, N, sort_indices.slice(0, j + 1));
                for (let k = 0; k < n_outliers; ++k) {
                    const index = n_i + n_j + k;
                    const out = samples[k];
                    triplets.set_entry(index, 0, i);
                    triplets.set_entry(index, 1, sim);
                    triplets.set_entry(index, 2, out);
                }
            }
        }
        return triplets;
    }

    /**
     * Should do the same as np.argsort()
     * @private
     * @param {Array} A 
     */
    __argsort(A) {
        return A
            .map((d, i) => {return {d: d, i: i};})
            .sort((a, b) => a.d - b.d)
            .map((d) => d.i);
    }

    /**
     * Samples {@link n_samples} integers from a given interval [0, {@link max_int}] while rejection the values that are in the {@link rejects}.
     * @private
     * @param {*} n_samples 
     * @param {*} max_int 
     * @param {*} rejects 
     */
    _rejection_sample(n_samples, max_int, rejects) {
        const randomizer = this.randomizer;
        const interval = linspace(0, max_int - 1).filter(d => rejects.indexOf(d) < 0);
        return randomizer.choice(interval, Math.min(n_samples, interval.length - 2));
    }

    /**
     * Calculates the weights for the sampled nearest neighbors triplets
     * @private
     * @param {Matrix} triplets - Sampled Triplets.
     * @param {Matrix} P - Pairwise similarity matrix.
     * @param {Matrix} nbrs - nearest Neighbors
     * @param {Float64Array} outlier_distances - Matrix of pairwise outlier distances
     * @param {Float64Array} sig - scaling factor for the distances.
     */
    _find_weights(triplets, P, nbrs, outlier_distances, sig) {
        const n_triplets = triplets.shape[0];
        const weights = new Float64Array(n_triplets);
        for (let t = 0; t < n_triplets; ++t) {
            const i = triplets.entry(t, 0);
            const sim = nbrs.row(i).indexOf(triplets.entry(t, 1));
            const p_sim = P.entry(i, sim);
            let p_out = Math.exp(-(outlier_distances[t] ** 2 / (sig[i] * sig[triplets.entry(t, 2)])));
            if (p_out < 1e-20) p_out = 1e-20;
            weights[t] = p_sim / p_out;
        }
        return weights;
    }

    /**
     * Sample uniformly ranom triplets
     * @private
     * @param {Matrix} X - Data matrix.
     * @param {Number} n_random - Number of random triplets per point
     * @param {Float64Array} sig - Scaling factor for the distances
     */
    _sample_random_triplets(X, n_random, sig) {
        const metric = this._metric;
        const randomizer = this.randomizer;
        const N = X.shape[0];
        const random_triplets = new Matrix(N * n_random, 3);
        const random_weights = new Float64Array(N * n_random);
        for (let i = 0; i < N; ++i) {
            const n_i = i * n_random;
            const indices = [...linspace(0, i - 1), ...linspace(i + 1, N - 1)];
            for (let j = 0; j < n_random; ++j) {
                let [sim, out] = randomizer.choice(indices, 2);
                let p_sim = Math.exp(-((metric(X.row(i), X.row(sim)) ** 2) / (sig[i] * sig[sim])));
                if (p_sim < 1e-20) p_sim = 1e-20;
                let p_out = Math.exp(-((metric(X.row(i), X.row(out)) ** 2) / (sig[i] * sig[out]))); 
                if (p_out < 1e-20) p_out = 1e-20;

                if (p_sim < p_out) {
                    [sim, out] = [out, sim];
                    [p_sim, p_out] = [p_out, p_sim];
                }
                const index = n_i + j;
                random_triplets.set_entry(index, 0, i);
                random_triplets.set_entry(index, 1, sim);
                random_triplets.set_entry(index, 2, out);
                random_weights[index] = p_sim / p_out;
            }
        }
        return {
            "random_triplets": random_triplets,
            "random_weights": random_weights,
        }
    }

    /**
     * Computes the gradient for updating the embedding.
     * @param {Matrix} Y - The embedding
     */
    _grad(Y) {
        const n_inliers = this.n_inliers;
        const n_outliers = this.n_outliers;
        const triplets = this.triplets;
        const weights = this.weights;
        const [N, dim] = Y.shape;
        const n_triplets = triplets.shape[0];
        const grad = new Matrix(N, dim, 0);
        let y_ij = new Array(dim).fill(0);
        let y_ik = new Array(dim).fill(0);
        let d_ij = 1;
        let d_ik = 1;
        let n_viol = 0;
        let loss = 0;
        const n_knn_triplets = N * n_inliers * n_outliers;

        for (let t = 0; t < n_triplets; ++t) {
            const [i, j, k] = triplets.row(t);
            // update y_ij, y_ik, d_ij, d_ik
            if (t % n_outliers == 0 || t >= n_knn_triplets) {
                d_ij = 1;
                d_ik = 1;
                for (let d = 0; d < dim; ++d) {
                    const Y_id = Y.entry(i, d);
                    const Y_jd = Y.entry(j, d);
                    const Y_kd = Y.entry(k, d);
                    y_ij[d] = Y_id - Y_jd;
                    y_ik[d] = Y_id - Y_kd;
                    d_ij += (y_ij[d] ** 2);
                    d_ik += (y_ik[d] ** 2);
                }
            // update y_ik and d_ik only
            } else {
                d_ik = 1;
                for (let d = 0; d < dim; ++d) {
                    const Y_id = Y.entry(i, d);
                    const Y_kd = Y.entry(k, d);
                    y_ik[d] = Y_id - Y_kd;
                    d_ik += (y_ik[d] ** 2);
                }
            }

            if (d_ij > d_ik) ++n_viol;
            loss += weights[t] / (1 + d_ik / d_ij);
            const w = (weights[t] / (d_ij + d_ik)) ** 2;
            for (let d = 0; d < dim; ++d) {
                const gs = y_ij[d] * d_ik * w;
                const go = y_ik[d] * d_ij * w;
                grad.set_entry(i, d, grad.entry(i, d) + gs - go);
                grad.set_entry(j, d, grad.entry(j, d) - gs);
                grad.set_entry(k, d, grad.entry(k, d) + go);
            }
        }
        return {
            "grad": grad,
            "loss": loss,
            "n_viol": n_viol,
        };
    }

    /**
     * 
     * @param {Number} max_iteration 
     */
    transform(max_iteration = 400) {
        for (let iter = 0; iter < max_iteration; ++iter) {
            this._next(iter);
        }
        return this.Y;
    }

    /**
     * @yields {Matrix}
     * @returns {Matrix}
     */
    * transform_iter() {
        for (let iter = 0; iter < 800; ++iter) {
            yield this._next(iter);
        }
        return this.Y;
    }

    /**
     * Does the iteration step.
     * @private
     * @param {Number} iter 
     */
    _next(iter) {
        const gamma = iter > 150 ? .5 : .3;
        const old_C = this.C;
        const vel = this.vel;
        const Y = this.Y.add(vel.mult(gamma));
        const {grad, loss, n_viol} = this._grad(Y);
        this.C = loss;
        this.Y = this._update_embedding(Y, iter, grad);
        this.lr *= (old_C > loss + this.tol)  ? 1.01 : .9;
        return this.Y;
    }

    /**
     * Updates the embedding.
     * @private
     * @param {Matrix} Y 
     * @param {Number} iter 
     * @param {Matrix} grad 
     */
    _update_embedding(Y, iter, grad) {
        const [N, dim] = Y.shape;
        const gamma = iter > 150 ? .9 : .5; // moment parameter
        const min_gain = .01;
        const gain = this.gain;
        const vel = this.vel;
        const lr = this.lr;
        for (let i = 0; i < N; ++i) {
            for (let d = 0; d < dim; ++d) {
                const new_gain = (Math.sign(vel.entry(i, d)) != Math.sign(grad.entry(i, d))) ? gain.entry(i, d) + .2 : Math.max(gain.entry(i, d) * .8, min_gain);
                gain.set_entry(i, d, new_gain);
                vel.set_entry(i, d, gamma * vel.entry(i, d) - lr * gain.entry(i, d) * grad.entry(i, d));
                Y.set_entry(i, d, Y.entry(i, d) + vel.entry(i, d));
            }
        }
        return Y;
    }
}

/**
 * @module dimensionality_reduction
 */

exports.Randomizer = Randomizer;
exports.kahan_sum = kahan_sum;
exports.neumair_sum = neumair_sum;
exports.euclidean = euclidean;
exports.euclidean_squared = euclidean_squared;
exports.cosine = cosine;
exports.manhattan = manhattan;
exports.chebyshev = chebyshev;
exports.k_nearest_neighbors = k_nearest_neighbors;
exports.distance_matrix = dmatrix;
exports.linspace = linspace;
exports.norm = norm;
exports.Matrix = Matrix;
exports.Reorder = Reorder;
exports.HNSW = HNSW;
exports.BallTree = BallTree;
exports.NNDescent = NNDescent;
exports.Heap = Heap;
exports.qr = qr;
exports.qr_householder = qr_householder;
exports.qr_givens = qr_givens;
exports.simultaneous_poweriteration = simultaneous_poweriteration;
exports.lu = lu;
exports.svrg = svrg;
exports.poweriteration_m = poweriteration_m;
exports.poweriteration_n = poweriteration_n;
exports.PCA = PCA;
exports.MDS = MDS;
exports.ISOMAP = ISOMAP;
exports.FASTMAP = FASTMAP;
exports.LDA = LDA;
exports.LLE = LLE;
exports.MLLE = MLLE;
exports.LTSA = LTSA;
exports.TSNE = TSNE;
exports.UMAP = UMAP;
exports.OAP = OAP;
exports.TriMap = TriMap;
exports.powell = powell;
exports.Hierarchical_Clustering = Hierarchical_Clustering;
exports.KMeans = KMeans;
exports.XMeans = XMeans;
exports.OPTICS = OPTICS;

Object.defineProperty(exports, '__esModule', { value: true });

}));
