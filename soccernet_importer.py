from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="./dataset")
mySoccerNetDownloader.password = 's0cc3rn3t'
mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["valid"])