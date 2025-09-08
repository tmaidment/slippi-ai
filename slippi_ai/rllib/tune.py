from absl import app, flags

from slippi_ai.rllib.experiment import Experiment

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment_name", None, "Name of the experiment")
flags.DEFINE_boolean("debug", False, "Debug mode flag")
flags.DEFINE_boolean("tune", False, "Tune mode flag")
flags.DEFINE_string("dolphin_path", None, "Path to Dolphin executable")
flags.DEFINE_string("iso_path", None, "Path to Melee ISO file")


def main(*args, **kwargs):
    print(f"Starting experiment {FLAGS.experiment_name}, Tuning: {FLAGS.tune}")
    Experiment(
        config={
            "debug": FLAGS.debug,
            "experiment_name": FLAGS.experiment_name,
            "tune": FLAGS.tune,
            "dolphin_path": FLAGS.dolphin_path,
            "iso_path": FLAGS.iso_path,
        }
    ).run()


if __name__ == "__main__":
    app.run(main)
