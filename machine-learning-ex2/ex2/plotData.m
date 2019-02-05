function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

  index1=find(y==1);
  index2=find(y==0);
  plot(X(index1,1),X(index1,2),'k+','markersize',7,'linewidth',2)
  plot(X(index2,1),X(index2,2),'ko','markerfacecolor','y','markersize',7);
%for index = 1 : length(y)  
  %if (y(index)==1)
   % plot(X(index,1),X(index,2),'k+','markersize',7,'linewidth',2);
  %else
   % plot(X(index,1),X(index,2),'ko','markerfacecolor','y','markersize',7);  
  %end
%end

% =========================================================================
hold off;

end
