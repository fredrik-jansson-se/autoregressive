use std::usize;

use autoregressive::univariate::Autoregressive;
use plotters::{coord::Shift, prelude::*};

const NUM_SAMPLES: usize = 1000;

fn plot<const N: usize>(
    drawing_area: DrawingArea<BitMapBackend, Shift>,
    caption: &str,
    params: &[f32; N],
) -> Result<(), Box<dyn std::error::Error>> {
    let ar = Autoregressive::new(0.0, 1.0, params);
    let data: Vec<(usize, f32)> = ar.enumerate().take(NUM_SAMPLES).collect();
    let (min, max) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(min, max), (_, v)| {
            (min.min(*v), max.max(*v))
        });
    let avg = data.iter().map(|(_, v)| v).sum::<f32>() / (data.len() as f32);

    let mut chart = ChartBuilder::on(&drawing_area)
        .caption(
            format!("{caption} min: {min:0.1} max: {max:0.1} avg: {avg:0.1}"),
            ("sans-serif", 20),
        )
        .margin_right(20)
        .margin_left(20)
        .margin_bottom(3)
        .build_cartesian_2d(0usize..NUM_SAMPLES, -7.0f32..7.0f32)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .max_light_lines(1)
        .draw()?;
    chart.draw_series(LineSeries::new(data.into_iter(), &BLUE))?;

    chart.configure_series_labels().draw()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let root = BitMapBackend::new("chart.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut areas = root.split_evenly((5, 1)).into_iter();

    plot(areas.next().unwrap(), "AR0", &[])?;
    plot(areas.next().unwrap(), "AR1 0.3", &[0.3])?;
    plot(areas.next().unwrap(), "AR1 0.9", &[0.9])?;
    plot(areas.next().unwrap(), "AR2 0.3 0.3", &[0.3, 0.3])?;
    plot(areas.next().unwrap(), "AR2 0.9 -0.8", &[0.9, -0.8])?;
    Ok(())
}
