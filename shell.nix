{
  # nixos-19.09-small
  pkgs ? import (fetchTarball https://github.com/NixOS/nixpkgs-channels/archive/02351ddb3a528383f7e42bc4b0eda9f2f631c525.tar.gz) {}
}:

let 
  r_pkgs = with pkgs.rPackages; [
    # rmarkdown related packages
    knitr
    rmarkdown
    # Rstudio related packages
    # servr
  ];
in
pkgs.mkShell {

  buildInputs = [
    # R packages
    pkgs.pandoc
    (pkgs.rWrapper.override {
      packages = r_pkgs;
    })

    # Java packages
    pkgs.openjdk11_headless
    pkgs.maven

    # Uncomment to add RStudio in your environment
    #(pkgs.rstudioWrapper.override {
    #  packages = r_pkgs;
    #})
  ];
}
