use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use tinbox::config::EngineConfig;
use tinbox::engine::backend::CandleBackend;
use tinbox::memory::kv_cache::{BlockManager, KVCachePool};
use tinbox::model::config::ModelConfig;
use tinbox::model::loader::SyntheticModelLoader;
use tinbox::runtime::engine::EngineRuntime;
use tinbox::runtime::scheduler::{Scheduler, SchedulerConfig};
use tinbox::serving::server::{create_router, AppState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("============================================================");
    info!(" Initializing TinBox: High-Throughput ML Runtime & Engine");
    info!("============================================================");

    let engine_config = EngineConfig::default();
    let model_config = ModelConfig::test_lfm3_config();
    let candle_device = candle_core::Device::Cpu;

    info!("Loading synthetic weights for model: {}", engine_config.model_id);
    let weights = SyntheticModelLoader::generate_synthetic_weights(&model_config, &candle_device)?;

    let kv_pool = KVCachePool::new(
        engine_config.num_gpu_blocks,
        model_config.num_hidden_layers,
        model_config.num_key_value_heads,
        engine_config.block_size,
        model_config.head_dim,
        candle_core::DType::F32,
        &candle_device,
    )?;

    let block_manager = BlockManager::new(engine_config.num_gpu_blocks, engine_config.block_size);
    let scheduler = Scheduler::new(
        SchedulerConfig {
            max_batch_size: engine_config.max_batch_size,
            max_num_batched_tokens: 4096,
        },
        block_manager,
    );

    let backend = CandleBackend::new(model_config, weights, candle_device);

    let (queue_tx, queue_rx) = mpsc::unbounded_channel();
    let runtime = EngineRuntime {
        scheduler,
        backend,
        kv_pool,
        rx_queue: queue_rx,
    };

    tokio::spawn(runtime.run_loop());

    let app_state = Arc::new(AppState { queue_tx });
    let app = create_router(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
    info!("TinBox server listening on http://0.0.0.0:8000");
    axum::serve(listener, app).await?;

    Ok(())
}
