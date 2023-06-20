s = sw_model('triAF', 1);

q_start = [0, 0, 0];

q_end = [1, 1, 0];

pts = 501;

spec = spinwave(s, {q_start, q_end, pts});
FMspec = sw_neutron(spec);
FMspec = sw_egrid(FMspec,'component','Sperp');

figure;
subplot(2,1,1)
sw_plotspec(FMspec,'mode',1,'colorbar',false)
axis([0 1 0 5])
subplot(2,1,2)
sw_plotspec(FMspec,'mode',2)
axis([0 1 0 2])
swplot.subfigure(1,3,1)

Sab = spec.Sab;
save("SAB.mat", "Sab");
