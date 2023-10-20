clear all;
x=128; y=128; z=64;
concat = zeros(x*2,y*2,z*2,3);
c=0;
for d =1:9
    j=0;
    if d~=2
        %if (d==11)
        %    j=j+64;
        %end
        c=c+1;
        for i=1:64
            im_name = sprintf('mask%d/%d.bmp', d, j+i+100);
            img = imread(im_name);
            r = mod(c-1,2);
            m = fix((c-1)/2);
            m4= fix((c-1)/4);
            if m4==1
                r = mod(c-5,2);
                m = fix((c-5)/2);
            end
            concat(1+m*y:y+m*y,1+r*x:x+r*x, i+m4*z,:)=img;
        end
    end
end


for k=1:128
    name = sprintf('256.256.128/mask1/%d.jpg', k+100);
    imwrite(uint8(reshape(concat(:,:,k,:), [x*2 y*2 3])), name);
end
