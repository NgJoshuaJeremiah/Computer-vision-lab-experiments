 function ret = map(img_l, img_r, s)
    n = s; 
    n1 = floor(n/2);
    [x,y]=size(img_l);
    ret = ones(x - n + 1, y - n + 1);

    msg = 'Creating Disparity Map ...';
    xw=0;
    fw = waitbar(xw, msg);

    for i = 1+n1 : x-n1
        for j = 1+n1 : y-n1
            cur_r = img_l(i-n1: i+n1, j-n1: j+n1);
            cur_l = rot90(cur_r, 2);
            min_coor = j; 
            min_diff = inf;

            % search for simmilar pattern in right image
            % limit search to 15 pixel to the left
            for k = max(1+n1 , j-14) : j
                T = img_r(i-n1: i+n1, k-n1: k+n1);
                cur_r = rot90(T, 2);

                % Calculate ssd and update minimum diff
                conv_1 = conv2(T, cur_r);
                conv_2 = conv2(T, cur_l);
                ssd = conv_1(n, n) - 2 * conv_2(n, n);
                if ssd < min_diff
                    min_diff = ssd;
                    min_coor = k;
                end
            end
            ret(i - n1, j - n1) = j - min_coor;
        end
        xw = i/(x-n1);
        waitbar(xw,fw);
    end
    close(fw);
 end