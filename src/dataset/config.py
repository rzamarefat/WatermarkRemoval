config = {
    "root_to_save_watermarked_images": "/home/rmarefat/projects/github/WatermarkRemoval/watermarked_images",
    "root_to_save_yolo_format_seg_boudning": "/home/rmarefat/projects/github/WatermarkRemoval/watermark_yolo_seg_bounding",

    
    # The following should be set to the dir which directly includes images - No subdir
    "root_to_read_original_images": "/home/rmarefat/projects/NSFW_zip_files_dataset/nudenet_predicted_gantman/training/neutral",

    "num_required_images": 1,
    "do_augmentation_on_bgs": True,


    "save_mask_images": True, #If this set is to True then root_to_save_watermarked_images_gt must be set to a valid empty dir
    "root_to_save_watermarked_images_gt": "/home/rmarefat/projects/github/WatermarkRemoval/watermarked_images_gt",

}
