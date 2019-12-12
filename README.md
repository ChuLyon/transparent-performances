# Cost of transparency

This repositery contains all elements used to evaluate the performances of
several classification systems.

These classification systems have been also evaluated on their degree
of transparency.

The aim of this experiment is to see if there is a correlation between
transparency degree and performances.

## Quick-start

To easily reproduce the results of this experiment you must install 
[nix](https://nixos.org/nix/)

```bash
curl https://nixos.org/nix/install | sh
```

Nix will allow you to reproduce this experiment in the extact same
dependencies than we used by entering the following command:

```bash
nix-shell --command "./runner.sh"
```

**This could takes several hours.**

If you simply want to plot our results, you can run the following command:

```bash
nix-shell --command "./renderer.R"
```

### Nix on Windows

Nix can be used on windows by using the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

### Running experiments without Nix

If Nix doesn't work, or you don't want to use it, you can manually install 
all dependencies described in the file *shell.nix* and run the script 
*runner.sh*.

However, you'll took the risk to reproduce experiments without the exact
same dependencies.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to 
discuss what you would like to change.

Please make sure to update tests as appropriate.

## Acknowledgment

This experiments was made in collaboration this civils hospitals of Lyon
with the aim to build transparent clinical decision support systems.

## License

The source code of this experiment is under the license
[GNU GPL3](https://www.gnu.org/licenses/gpl-3.0.en.html)
