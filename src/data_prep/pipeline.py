import os
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    MinhashDedupSignature,
    MinhashDedupFilter,
    MinhashDedupCluster
)
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.hashing import HashConfig

from src.data_prep.filters import SpamLogCyclicFilter
from datatrove.pipeline.filters import FastTextClassifierFilter, FineWebQualityFilter


def run_data_prep_pipeline(cfg):
    input_path = cfg.get("input_path", "./data/raw")
    output_path = cfg.get("output_path", "./data/deduplicated")
    mh_base = cfg.get("minhash_base_path", "./data/minhash")

    mh_cfg = cfg.get("minhash_config", {})
    n_grams = mh_cfg.get("n_grams", 5)
    num_buckets = mh_cfg.get("num_buckets", 14)
    hashes_per_bucket = mh_cfg.get("hashes_per_bucket", 8)
    precision = mh_cfg.get("precision", 64)

    filters_cfg = cfg.get("filters", {})
    remove_seo = filters_cfg.get("remove_seo", True)
    remove_logs = filters_cfg.get("remove_logs", True)
    remove_cyclic = filters_cfg.get("remove_cyclic", True)
    fasttext_spam_cfg = filters_cfg.get("fasttext_spam", {})
    fineweb_quality_cfg = filters_cfg.get("fineweb_quality", {})

    # Common Config
    config = MinhashConfig(
        hash_config=HashConfig(precision=precision),
        num_buckets=num_buckets,
        hashes_per_bucket=hashes_per_bucket,
        n_grams=n_grams
    )

    INPUT_READER = JsonlReader(input_path)
    TOTAL_TASKS = 1 # local simple execution


    stage1_pipeline = [
        INPUT_READER,
        SpamLogCyclicFilter(
            remove_seo=remove_seo,
            remove_logs=remove_logs,
            remove_cyclic=remove_cyclic,
            exclusion_writer=JsonlWriter(os.path.join(output_path, "removed_spam_logs"))
        )
    ]

    if fasttext_spam_cfg.get("enabled", False):
        model_url = fasttext_spam_cfg.get("model_url", "hf://datatrove/fasttext-spam/model.bin")
        stage1_pipeline.append(
            FastTextClassifierFilter(
                model_url=model_url,
                exclusion_writer=JsonlWriter(os.path.join(output_path, "removed_fasttext_spam"))
            )
        )

    if fineweb_quality_cfg.get("enabled", False):
        stage1_pipeline.append(
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(os.path.join(output_path, "removed_fineweb_quality"))
            )
        )

    stage1_pipeline.append(
        MinhashDedupSignature(
            output_folder=os.path.join(mh_base, "signatures"),
            config=config
        )
    )

    # 1. Custom Filter + Signatures
    stage1 = LocalPipelineExecutor(
        pipeline=stage1_pipeline,
        tasks=TOTAL_TASKS
    )


    # 2. Buckets
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=os.path.join(mh_base, "signatures"),
                output_folder=os.path.join(mh_base, "buckets"),
                config=config,
                only_dedup_in_index=False
            )
        ],
        tasks=config.num_buckets
    )

    # 3. Cluster
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=os.path.join(mh_base, "buckets"),
                output_folder=os.path.join(mh_base, "clusters"),
                config=config,
                save_cluster_id=True,
                save_cluster_size=True
            )
        ],
        tasks=1
    )

    # 4. Filter
    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(
                input_folder=os.path.join(mh_base, "clusters"),
                exclusion_writer=JsonlWriter(os.path.join(output_path, "removed_duplicates"))
            ),
            JsonlWriter(os.path.join(output_path, "final"))
        ],
        tasks=TOTAL_TASKS
    )

    print("Running Stage 1: Filters & Signatures...")
    stage1.run()

    print("Running Stage 2: Buckets...")
    stage2.run()

    print("Running Stage 3: Cluster...")
    stage3.run()

    print("Running Stage 4: Filter Duplicates...")
    stage4.run()

    print(f"Data Prep Pipeline Completed. Deduplicated output at: {os.path.join(output_path, 'final')}")
