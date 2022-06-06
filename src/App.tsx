import React, { useEffect, useState } from 'react';
import './App.css';
import Button from "@mui/material/Button";
import { Alert, AppBar, Box, Card, CardActions, CardContent, CircularProgress, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Grid, Stack, Toolbar, Typography } from '@mui/material';
import * as tf from "@tensorflow/tfjs";
import { GraphModel, Rank, Tensor } from '@tensorflow/tfjs';
import Dropzone from 'react-dropzone';

const modelUrl = "https://storage.googleapis.com/covid19-ct-scan-models/model.json";

const App = () => {
  const [image, setImage] = useState("");
  const [cnnModel, setCnnModel] = useState<GraphModel>();
  const [loadModel, setLoadModel] = useState(false);
  const [open, setOpen] = useState(false);
  const [hasCovid, setHasCovid] = useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  }

  const handleClose = () => {
    setOpen(false);
  }

  useEffect(() => {
    tf.ready().then(() => {
      tf.loadGraphModel(modelUrl).then(model => {
        setCnnModel(model)
        setLoadModel(true);
        console.log("Successfully loaded cnn model");
        
      });
    });
  }, []);

  return (
    <div className="App">
      <Stack spacing={10}>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar>
            <Toolbar>
              <Typography>COVID-19 diagnosis</Typography>
            </Toolbar>
          </AppBar>
        </Box>
        <Grid container justifyContent="center">
          <Card elevation={8} sx={{ minWidth: 400 }}>
            <CardContent>
              {loadModel ? <Grid container direction={"column"} alignItems="center">
                <Stack spacing={10}>
                  <Typography variant='h5' gutterBottom component="div">COVID-19 CT-SCAN Classifier</Typography>
                </Stack>
                <Dropzone onDrop={acceptedFiles => {
                  setImage(URL.createObjectURL(acceptedFiles[0]))
                }}>
                  {({ getRootProps, getInputProps }) => (
                    <section>
                      <div {...getRootProps()}>
                        <input {...getInputProps()} />
                        <p>Drag 'n' drop image, or click to select image</p>
                        {image && <img id='ct-scan-image' width={256} height={256} src={image} alt="image" />}
                      </div>
                    </section>
                  )}
                </Dropzone>
              </Grid> : <CircularProgress />}
            </CardContent>
            <CardActions>
              <Grid container direction={"column"} alignItems="center">
                <Button disabled={image ? false : true} style={{ marginLeft: 10 }} size='small' variant="contained" onClick={() => {
                  const ctScan = document.getElementById("ct-scan-image") as HTMLImageElement;
                  const img = tf.browser.fromPixels(ctScan).mean(2).toFloat().expandDims(0).expandDims(-1);
                  const prediction = cnnModel?.predict(img) as Tensor<Rank>;
                  prediction.data().then(data => {
                    console.log(prediction.toString())
                    if (data[0] < data[1]) {
                      setHasCovid(false);
                    } else {
                      setHasCovid(true);
                    }
                    handleClickOpen();
                  });
                }}>Predict</Button>
              </Grid>
            </CardActions>
          </Card>
        </Grid>
        <Dialog
          open={open}
          onClose={handleClose}
          aria-labelledby="alert-dialog-title"
          aria-describedby="alert-dialog-description"
        >
          <DialogTitle id="alert-dialog-title">
            {"Results from diagnostic"}
          </DialogTitle>
          <DialogContent>
            <DialogContentText id="alert-dialog-description">
              {hasCovid ? <Alert severity="error">Patient has COVID-19, please take take the necessary precautions and visit a doctor.</Alert> : <Alert severity="info">Patient is Healthy.</Alert>}
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleClose}>Ok</Button>
          </DialogActions>
        </Dialog>
      </Stack>
    </div>
  );
}

export default App;

