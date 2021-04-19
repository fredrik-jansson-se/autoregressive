use plotters::prelude::*;

fn main() {
    let root = BitMapBackend::new("chart.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(0usize..100usize, 0.0f32..80.0f32)
        .unwrap();
    chart.configure_mesh().x_labels(10).draw().unwrap();

    let mut ar = autoregressive::univariate::Autoregressive::new(5.0, 1.0, &[0.5]);
    chart
        .draw_series(LineSeries::new((0..100).map(|x| (x, ar.step())), &RED))
        .unwrap();
}
