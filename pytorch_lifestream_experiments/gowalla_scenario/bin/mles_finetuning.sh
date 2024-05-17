python -m ptls.pl_train_module \
    trainer.max_epochs=1 \
    model_path="models/mles_model.p" \
    --config-dir conf --config-name mles_params

python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_mles