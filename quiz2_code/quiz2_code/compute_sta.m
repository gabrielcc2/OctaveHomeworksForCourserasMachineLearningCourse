function [ sta ] = compute_sta( stim, rho, num_timesteps )
%COMPUTE_STA Calculates the spike-triggered average for a neuron that
%            is driven by a stimulus defined in stim. The spike-
%            triggered average is computed over num_timesteps timesteps.
    sta = zeros(num_timesteps, 1);

    % This command finds the indices of all of the spikes that occur
    % after 300 ms into the recording.
    spike_times = find(rho(num_timesteps+1:100)) + num_timesteps;


%deberia ser end donde dice 10000

    % Fill in this value. Note that you should not count spikes that occur
    % before 300 ms into the recording.
    num_spikes = size(spike_times)(1);
    
    
    % Compute the spike-triggered average of the spikes found using the
    % find command. To do this, compute the average of all of the vectors
    % starting 300 ms (exclusive) before a spike and ending at the time of
    % the event (inclusive). Each of these vectors defines a list of
    % samples that is contained within a window of 300 ms before the each
    % spike. The average of these vectors should be completed in an
    % element-wise manner.
    % 
    % Your code goes here.
    for time_sample=1:num_timesteps
      for visited_spike=1:num_spikes
        sta(time_sample,1)=sta(time_sample,1)+stim((spike_times(visited_spike,1)-150+time_sample),1)
      endfor
      sta(time_sample,1)=sta(time_sample,1)/num_spikes;
    endfor

end
