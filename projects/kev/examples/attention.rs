use kev::example;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = example::INPUTS;
    let mut head = example::head();
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [] => {}
        [flag] if flag == "--uniform" => head.query_weights = [[0.0; 2]; 2],
        _ => return Err("usage: cargo run --example attention -- [--uniform]".into()),
    }

    let trace = head.forward(&inputs)?;
    println!("Kev: one self-attention head (hand-set weights, no training yet)");
    println!("Inputs: {:?}", inputs);
    println!("Queries: {:?}", trace.queries);
    println!("Keys: {:?}", trace.keys);
    println!("Values: {:?}", trace.values);
    for index in 0..inputs.len() {
        println!("\nToken {index}");
        println!("  scores:  {:.6?}", trace.scores[index]);
        println!("  weights: {:.6?}", trace.weights[index]);
        println!("  output:  {:.6?}", trace.output[index]);
    }
    Ok(())
}
