%% Create All Permutations
n = de2bi(0:255);

%% Check if Simple
s(:,1) = CheckSimpleNorth(n);
s(:,2) = CheckSimpleEast(n);
s(:,3) = CheckSimpleSouth(n);
s(:,4) = CheckSimpleWest(n);
