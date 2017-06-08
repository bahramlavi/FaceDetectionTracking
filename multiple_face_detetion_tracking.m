
function multiple_face_detetion_tracking()

profile on
clear all;
clc
% Create system faces used for reading video, detecting moving faces,
% and displaying the results.
obj = initializeObjects();

tracks = initializeTracks(); % Create an empty array of tracks.

nextId = 1; % ID of the next track


while(1)
    frame = readFrame();   
    bboxes = detectObjects(frame);    
    TracksByPredict();
    [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment();
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    addNewTracks();
    displayTrackingResults();
end

function obj = initializeObjects()

        obj.reader = imaq.VideoDevice();
        set(obj.reader,'ReturnedColorSpace','rgb');
        obj.videoPlayer = vision.VideoPlayer();
                   
        obj.blobAnalyser = vision.CascadeObjectDetector;      
        obj.blobAnalyser.MinSize=[20 20];                        
        
end

 function tracks = initializeTracks()
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount',{},...
            'DTime',{});
 end
 function frame = readFrame()
        frame = obj.reader.step();
 end
function bboxes = detectObjects(frame) 
    
        bboxes = obj.blobAnalyser.step(frame);   
        
 end
  function TracksByPredict()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;           

            predictedCentroid = predict(tracks(i).kalmanFilter);
            predictedCentroid = predictedCentroid - bbox(1:4) / 2;

            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
  end
 function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        nTracks = length(tracks);
        nDetections = size(bboxes, 1);
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, bboxes);
        end
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
 end

 function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            
            
            bbox = bboxes(detectionIdx, :);
            if ~isempty(bbox)
                %bbox=predict(tracks(trackIdx).kalmanFilter);
                bbox=correct(tracks(trackIdx).kalmanFilter, bbox);
            else
                 bbox= predict(tracks(trackIdx).kalmanFilter);
            end
            tracks(trackIdx).bbox = bbox;

            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;            
                        
        end
 end
 function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = tracks(ind).consecutiveInvisibleCount + 1;
        end
 end
   function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        invisibleForTooLong = 50;
        ageThreshold = 500;

        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        lostInds = (ages < ageThreshold & visibility < 1) | [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;          
        tracks = tracks(~lostInds);
   end
    
        
   function addNewTracks()
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(bboxes, 1)
            tic
            bbox = bboxes(i, :);
            
            kalmanFilter = configureKalmanFilter('ConstantAcceleration', bbox, [50, 50,50], [250,100,100], 500);                        
            %kalmanFilter = configureKalmanFilter('ConstantAcceleration', bbox, [250 250 250], [250,100,100], 12500);                        
            
            dtime=toc;
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount',0,...
                'DTime',dtime);
                       
            tracks(end + 1) = newTrack;

            nextId = nextId + 1;
        end
   end
function displayTrackingResults()                     
            
        minVisibleCount = 1;
        
        if ~isempty(tracks)

            reliableTrackInds = [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);          

            if ~isempty(reliableTracks) %&& reliableTracks.islost==0                                
                    bboxes = cat(1, reliableTracks.bbox);                    
                    ids = int32([reliableTracks(:).id]);                              
                    labels = cellstr(int2str(ids'));
                    predictedTrackInds =[reliableTracks(:).consecutiveInvisibleCount] > 0;
                    isPredicted = cell(size(labels));
                    isPredicted(predictedTrackInds) = {'predicted'};
                    labels = strcat(labels, isPredicted);
                    %detectTimes=[reliableTracks(:).DTime];
                    %timeLables=cellstr(num2str(detectTimes'));
                    %timeLables=strcat('DT:',timeLables');
                    frame = insertObjectAnnotation(frame, 'rectangle',bboxes, 'Face','Color','red','TextColor','black');   
                    %AvgDTime=mean(detectTimes);
                    %AvgDTime=cellstr(num2str(AvgDTime));
                    %AvgDTime=strcat('Mean Value along Detection Times: ',AvgDTime);
                    %frame= insertText(frame,[30 475],AvgDTime,'AnchorPoint','LeftBottom','BoxOpacity', 0.6,'BoxColor','Green');
                    %disp(mean(detectTimes))
                    %frame = insertShape(frame, 'rectangle',bboxes,'color','red');    
                    %frame=insertText(frame,bboxes(1:2),reliableTracks.DTime,'AnchorPoint','LeftBottom');
            end
       end  
        frame = im2uint8(frame); 
        obj.videoPlayer.step(frame);     
        
end

profile report
end







