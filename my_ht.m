function h = my_ht(t)
    h = exp(-3 .* t) .* unitstep(t);
end