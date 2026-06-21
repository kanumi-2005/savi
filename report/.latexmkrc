@default_files = ("report.tex");
$pdf_mode = 4;
$aux_dir = ".aux";

add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
sub run_makeglossaries {
    system "makeglossaries -d $aux_dir $root_filename";
}
