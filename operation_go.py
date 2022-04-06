import do_sample as do
import send_to_git as stg

"""
Denne filen kalles fra cron-table i RPi hver sjette time (definert tid).
"""

do.sample()
stg.run()
