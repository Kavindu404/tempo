# Save final model
    final_model_path = os.path.join(output_dir, "sam2_finetuned_final.pth")
    torch.save({
        'iteration': args.iterations - 1,
        'model_state_dict': predictor.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if 'loss' in locals() else None,
        'mean_iou': mean_iou,
        'best_ap': best_ap
    }, final_model_path)
    
    # Plot training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    logger.info(f"Training completed. Final model saved at {final_model_path}")
    logger.info(f"Final Mean IoU: {mean_iou:.4f}, Best AP: {best_ap:.4f}")
    
    # Final validation
    final_val_dir = os.path.join(output_dir, "final_validation")
    os.makedirs(final_val_dir, exist_ok=True)
    
    # Run validation on final model
    with torch.no_grad():
        predictor.model.eval()
        predictions, pred_file = create_coco_predictions(predictor, val_data, final_val_dir)
        final_metrics = evaluate_coco(coco_gt, pred_file)
    
    # Log final validation results
    logger.info("Final validation results:")
    for metric, value in final_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save training config and results summary
    summary = {
        'config': vars(args),
        'best_ap': best_ap,
        'final_mean_iou': float(mean_iou),
        'final_metrics': final_metrics,
        'dataset_stats': {
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        },
        'training_time': time.time() - start_time
    }
    
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return best_ap


if __name__ == "__main__":
    args = parse_args()
    main(args)
