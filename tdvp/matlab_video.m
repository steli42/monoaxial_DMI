clear;
clc;

% Define the folder where the PNG files are located
folder = 'out/series'; % Adjust this to your folder path

% Get a list of all PNG files in the folder
pngFiles = dir(fullfile(folder, '*.png'));

% Sort the PNG files by their names
[~, idx] = sort({pngFiles.name});
pngFiles = pngFiles(idx);

% Create a video writer object
videoFile = 'video.mp4';
video = VideoWriter(videoFile, 'MPEG-4');

% Set the desired frame rate (e.g., 6 fps for slower playback)
video.FrameRate = 30; % Adjust this value as needed

open(video);


% Optionally, add each image to the video in forward order
for i = 1:length(pngFiles)
    imageName = fullfile(folder, pngFiles(i).name); % Ensure correct path to the image
    if exist(imageName, 'file') % Check if the file exists
        img = imread(imageName);
        writeVideo(video, img);
    else
        warning('Image file not found: %s', imageName);
    end
end

close(video);

disp('Video created successfully.');
