function display_matrix(I, ax)
% Display matrix where each column is a vectorised square image
if ~exist('ax', 'var')
    ax = gca;
end

[L2, M2] = size(I);
L = sqrt(L2); % each image is L x L
M = ceil(sqrt(M2)); % there are M2 images (M2 <= M x M)

I = I ./ max(abs(I)); % normalise
I(isnan(I)) = 0;
I = reshape(I, L, L, M2);
B = round(L / 10);

J = ones( M*(L+B)+B ) * 1.02;

for i = 1 : M % columns
    for j = 1 : M % rows
        n = (i-1)*M+j;
        if n > M2
            break
        end
        J( (B+L)*i-L+(1:L), (B+L)*j-L+(1:L) ) = I(:,:,n);
    end
end

imagesc(ax, J, 'AlphaData', double(~isnan(J)));
colormap(ax, 'gray');
axis(ax, 'image');
axis(ax, 'off');
end