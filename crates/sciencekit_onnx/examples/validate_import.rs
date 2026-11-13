//! Import every fixture model and write its predictions for the harness.
//!
//! Usage: `validate_import <fixture-root>`; each fixture directory gets an
//! `imported.json` beside its `model.onnx` and `input.json`.

use std::collections::BTreeMap;
use std::path::Path;

use sciencekit_onnx::SKOnnxPredictor;
use serde_json::{Value, json};

fn main() {
    let root = std::env::args().nth(1).expect("a fixture root argument");
    let root = Path::new(&root);
    let mut directories: Vec<_> = std::fs::read_dir(root)
        .expect("the fixture root is readable")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.join("model.onnx").exists())
        .collect();
    directories.sort();
    let mut matched = 0usize;
    for directory in &directories {
        match import_one(directory) {
            Ok(()) => {
                matched += 1;
                println!(
                    "IMPORT {}: ok",
                    directory.file_name().unwrap().to_string_lossy()
                );
            }
            Err(error) => println!(
                "IMPORT {}: FAIL {error}",
                directory.file_name().unwrap().to_string_lossy()
            ),
        }
    }
    println!("{matched}/{} models imported", directories.len());
}

fn import_one(directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let input: Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("input.json"))?)?;
    let rows: Vec<Vec<f64>> = input["input"]
        .as_array()
        .ok_or("input.json has no matrix")?
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .map(|cell| cell.as_f64().unwrap_or(0.0))
                .collect()
        })
        .collect();
    let height = rows.len();
    let width = rows.first().map(Vec::len).unwrap_or(0);
    let features = ndarray::Array2::from_shape_vec((height, width), rows.concat())?;
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model_file(directory.join("model.onnx"))
        .build()?;
    let outputs = predictor.predict_arrays(&features)?;
    let mut serialized = BTreeMap::new();
    for (name, array) in outputs {
        let shape: Vec<usize> = array.shape().to_vec();
        let data: Vec<f64> = array.iter().copied().collect();
        serialized.insert(name, json!({"shape": shape, "data": data}));
    }
    std::fs::write(
        directory.join("imported.json"),
        json!({"outputs": serialized}).to_string(),
    )?;
    Ok(())
}
