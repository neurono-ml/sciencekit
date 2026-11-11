//! Pairwise coupling multiclass probability solver (Wu, Lin & Weng 2004)
//! (spec `support-vector-machines`).

use sciencekit_common::SKFloat;

/// Couple binary probabilities `P(y = b | y ∈ {a, b})` (`a < b`) with an
/// explicit tolerance and iteration cap, returning `(probabilities, iterations, converged)`.
pub fn sk_couple_pairwise_probabilities_with_budget<F: SKFloat>(
    pairwise_positive_probabilities: &[F],
    number_of_classes: usize,
    tolerance: f64,
    maximum_iterations: usize,
) -> (Vec<F>, usize, bool) {
    if number_of_classes == 2 {
        let p_pos = pairwise_positive_probabilities[0]
            .to_f64()
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        return (
            vec![
                F::from(1.0 - p_pos).expect("fits SKFloat"),
                F::from(p_pos).expect("fits SKFloat"),
            ],
            0,
            true,
        );
    }

    let min_prob = 1e-7_f64;
    let mut r = vec![vec![0.0_f64; number_of_classes]; number_of_classes];
    let mut cursor = 0usize;
    for class_a in 0..number_of_classes {
        for class_b in (class_a + 1)..number_of_classes {
            let p_b = pairwise_positive_probabilities[cursor]
                .to_f64()
                .unwrap_or(0.5)
                .clamp(min_prob, 1.0 - min_prob);
            r[class_b][class_a] = p_b;
            r[class_a][class_b] = 1.0 - p_b;
            cursor += 1;
        }
    }

    let mut q = vec![vec![0.0_f64; number_of_classes]; number_of_classes];
    for t in 0..number_of_classes {
        let mut diag = 0.0_f64;
        for s in 0..number_of_classes {
            if s != t {
                diag += r[s][t] * r[s][t];
                q[t][s] = -r[t][s] * r[s][t];
            }
        }
        q[t][t] = diag;
    }

    let mut p = vec![1.0_f64 / (number_of_classes as f64); number_of_classes];
    let mut qp = vec![0.0_f64; number_of_classes];
    for t in 0..number_of_classes {
        qp[t] = (0..number_of_classes).map(|s| q[t][s] * p[s]).sum();
    }

    let mut executed = 0usize;
    let mut converged = false;
    while executed < maximum_iterations {
        let p_qp: f64 = (0..number_of_classes).map(|t| p[t] * qp[t]).sum();
        let max_error = (0..number_of_classes)
            .map(|t| (qp[t] - p_qp).abs())
            .fold(0.0_f64, f64::max);
        if max_error < tolerance {
            converged = true;
            break;
        }
        executed += 1;

        let mut running_p_qp = p_qp;
        for t in 0..number_of_classes {
            let diff = (-qp[t] + running_p_qp) / q[t][t];
            p[t] += diff;
            running_p_qp = (running_p_qp + diff * (diff * q[t][t] + 2.0 * qp[t]))
                / ((1.0 + diff) * (1.0 + diff));
            let scale = 1.0 / (1.0 + diff);
            for s in 0..number_of_classes {
                qp[s] = (qp[s] + diff * q[t][s]) * scale;
                p[s] *= scale;
            }
        }
    }

    let sum: f64 = p.iter().map(|v| v.max(0.0)).sum();
    let normalized: Vec<F> = if sum > 0.0 {
        p.into_iter()
            .map(|v| F::from((v.max(0.0) / sum).clamp(0.0, 1.0)).expect("fits SKFloat"))
            .collect()
    } else {
        vec![
            F::from(1.0 / (number_of_classes as f64)).expect("fits SKFloat");
            number_of_classes
        ]
    };
    (normalized, executed, converged)
}

/// Couple binary probabilities using the standard tolerance `1e-5 / c` and
/// iteration cap `max(100, c)`.
pub fn sk_couple_pairwise_probabilities<F: SKFloat>(
    pairwise_positive_probabilities: &[F],
    number_of_classes: usize,
) -> Vec<F> {
    let tolerance = 1e-5 / (number_of_classes.max(1) as f64);
    let maximum_iterations = 100.max(number_of_classes);
    sk_couple_pairwise_probabilities_with_budget(
        pairwise_positive_probabilities,
        number_of_classes,
        tolerance,
        maximum_iterations,
    )
    .0
}
