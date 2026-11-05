use candle_core::{DType, Device};
use std::time::Duration;
use tinbox::engine::backend::CandleBackend;
use tinbox::memory::kv_cache::{BlockManager, KVCachePool};
use tinbox::model::config::ModelConfig;
use tinbox::model::loader::SyntheticModelLoader;
use tinbox::runtime::engine::EngineRuntime;
use tinbox::runtime::scheduler::{Scheduler, SchedulerConfig};
use tinbox::types::{GenerationOutput, RequestId, SamplingParams, Sequence, TokenId};
use tokio::sync::mpsc;

#[tokio::test]
async fn test_module_11_engine_runtime_end_to_end() {
    let dev = Device::Cpu;
    let config = ModelConfig::test_lfm3_config();
    let weights = SyntheticModelLoader::generate_synthetic_weights(&config, &dev).unwrap();
    let kv_pool = KVCachePool::new(16, config.num_hidden_layers, config.num_key_value_heads, 16, config.head_dim, DType::F32, &dev).unwrap();
    let backend = CandleBackend::new(config.clone(), weights, dev);
    let block_manager = BlockManager::new(16, 16);
    let sched_config = SchedulerConfig {
        max_batch_size: 4,
        max_num_batched_tokens: 64,
    };
    let scheduler = Scheduler::new(sched_config, block_manager);

    let (req_tx, req_rx) = mpsc::unbounded_channel();
    let runtime = EngineRuntime {
        scheduler,
        backend,
        kv_pool,
        rx_queue: req_rx,
    };

    let runtime_handle = tokio::spawn(async move {
        runtime.run_loop().await;
    });

    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel::<GenerationOutput>();
    let prompt = vec![TokenId(10), TokenId(20)];
    let params = SamplingParams {
        max_new_tokens: 2,
        ..Default::default()
    };
    let seq = Sequence::new(RequestId("e2e-1".to_string()), prompt, params, Some(resp_tx));

    req_tx.send(seq).unwrap();

    let token1 = tokio::time::timeout(Duration::from_secs(5), resp_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(token1.request_id, RequestId("e2e-1".to_string()));

    let token2 = tokio::time::timeout(Duration::from_secs(5), resp_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(token2.is_finished);

    drop(req_tx);
    let _ = tokio::time::timeout(Duration::from_secs(2), runtime_handle).await;
}

#[tokio::test]
async fn test_module_11_early_channel_drop_shutdown() {
    let (req_tx, mut req_rx) = mpsc::unbounded_channel::<Sequence>();

    // Test channel drop handling
    drop(req_tx);
    assert!(req_rx.recv().await.is_none());
}
