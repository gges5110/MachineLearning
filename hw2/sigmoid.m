function [ Z ] = sigmoid( Y )
Z = 1 ./ ( 1 + exp(-Y) );
end

