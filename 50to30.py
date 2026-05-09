import os
import shutil
import math


def convert_fps():
    # Configuration
    input_dir = './frames'
    output_dir = './frame30'
    source_fps = 50
    target_fps = 30
    duration_sec = 10

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of all files and sort them numerically
    # This ensures frame 2 comes after frame 1, not frame 10
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # logic to sort files like 1.png, 2.png ... 10.png correctly
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    total_source_frames = len(files)
    target_frame_count = duration_sec * target_fps  # Should be 300

    print(f"Source: {total_source_frames} frames")
    print(f"Target: {target_frame_count} frames")

    for i in range(target_frame_count):
        # Mathematical formula to pick the frame
        # frame_index = floor( target_index * (source_fps / target_fps) )
        source_index = math.floor(i * (source_fps / target_fps))

        # Safety check to ensure we don't go out of bounds
        if source_index >= total_source_frames:
            source_index = total_source_frames - 1

        # Get the filename of the source frame
        source_filename = files[source_index]

        # Create new filename (e.g., frame_000.png)
        target_filename = f"frame_{i:03d}.png"

        src_path = os.path.join(input_dir, source_filename)
        dst_path = os.path.join(output_dir, target_filename)

        # Copy the file
        shutil.copy2(src_path, dst_path)

        # Optional: Print progress every 50 frames
        if i % 50 == 0:
            print(f"Processed output frame {i}/{target_frame_count} (Source frame: {source_index})")

    print("Conversion complete. Check ./frame30 folder.")


if __name__ == "__main__":
    convert_fps()