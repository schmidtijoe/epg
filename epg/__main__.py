"""
Programm to calculate EPG formalism following Weigel et al. 2015,
Originally based on https://github.com/imr-framework/epg

- added package functionality
- refactored for using exact sequence timings rfs and grads
- adopted non integer grad moment shifts
________________________________
Jochen Schmidt, 19.01.2023
"""
import logging
import options as opt
import sequence
import epg_sim

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('simple_parsing.wrappers').disabled = True
logging.getLogger('simple_parsing.help_formatter').disabled = True


def main(prog_opts: opt.Config):
    semc_seq = sequence.create_semc(params=prog_opts.params)
    epg = epg_sim.EPG(semc_seq)
    logging.info("Start Simulation")
    epg.simulate()
    logging.info("Finished \nPlotting")
    epg.plot_epg(save='test/epg_nonint_crusher_shifts.png')
    epg.plot_echoes(save='test/echos_nonint_crusher_shifts.png')


if __name__ == '__main__':
    # get cmd line arguments
    parser, args = opt.create_cmd_parser()
    opts = opt.Config.from_cmd_args(args)

    # set logger
    if opts.config.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=log_level)

    # start prog
    try:
        main(opts)
    except (ValueError, AttributeError) as e:
        logging.error(e)
        parser.print_usage()
