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
        let positive_probability = pairwise_positive_probabilities[0]
            .to_f64()
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        return (
            vec![
                F::from(1.0 - positive_probability).expect("fits SKFloat"),
                F::from(positive_probability).expect("fits SKFloat"),
            ],
            0,
            true,
        );
    }

    let minimum_probability = 1e-7_f64;
    let mut r = vec![vec![0.0_f64; number_of_classes]; number_of_classes];
    let mut cursor = 0usize;
    for class_a in 0..number_of_classes {
        for class_b in (class_a + 1)..number_of_classes {
            let positive_probability = pairwise_positive_probabilities[cursor]
                .to_f64()
                .unwrap_or(0.5)
                .clamp(minimum_probability, 1.0 - minimum_probability);
            r[class_b][class_a] = positive_probability;
            r[class_a][class_b] = 1.0 - positive_probability;
            cursor += 1;
        }
    }

    let q: Vec<Vec<f64>> = (0..number_of_classes)
        .map(|row_class| {
            let diagonal: f64 = r
                .iter()
                .enumerate()
                .filter(|&(other_class, _)| other_class != row_class)
                .map(|(_, other_row)| other_row[row_class] * other_row[row_class])
                .sum();
            (0..number_of_classes)
                .map(|column_class| {
                    if column_class == row_class {
                        diagonal
                    } else {
                        -r[row_class][column_class] * r[column_class][row_class]
                    }
                })
                .collect()
        })
        .collect();

    let mut p = vec![1.0_f64 / (number_of_classes as f64); number_of_classes];
    let mut qp: Vec<f64> = q
        .iter()
        .map(|row_q| {
            row_q
                .iter()
                .zip(p.iter())
                .map(|(&q_entry, &probability)| q_entry * probability)
                .sum()
        })
        .collect();

    let mut executed = 0usize;
    let mut converged = false;
    while executed < maximum_iterations {
        let p_qp: f64 = p
            .iter()
            .zip(qp.iter())
            .map(|(&probability, &qp_entry)| probability * qp_entry)
            .sum();
        let maximum_error = qp
            .iter()
            .map(|&qp_entry| (qp_entry - p_qp).abs())
            .fold(0.0_f64, f64::max);
        if maximum_error < tolerance {
            converged = true;
            break;
        }
        executed += 1;

        let mut running_p_qp = p_qp;
        for (row_class, row_q) in q.iter().enumerate() {
            let diagonal = row_q[row_class];
            let difference = (-qp[row_class] + running_p_qp) / diagonal;
            p[row_class] += difference;
            running_p_qp = (running_p_qp
                + difference * (difference * diagonal + 2.0 * qp[row_class]))
                / ((1.0 + difference) * (1.0 + difference));
            let scale = 1.0 / (1.0 + difference);
            qp.iter_mut().zip(p.iter_mut()).zip(row_q.iter()).for_each(
                |((qp_entry, probability), &q_entry)| {
                    *qp_entry = (*qp_entry + difference * q_entry) * scale;
                    *probability *= scale;
                },
            );
        }
    }

    let sum: f64 = p.iter().map(|value| value.max(0.0)).sum();
    let normalized: Vec<F> = if sum > 0.0 {
        p.into_iter()
            .map(|value| F::from((value.max(0.0) / sum).clamp(0.0, 1.0)).expect("fits SKFloat"))
            .collect()
    } else {
        vec![F::from(1.0 / (number_of_classes as f64)).expect("fits SKFloat"); number_of_classes]
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
