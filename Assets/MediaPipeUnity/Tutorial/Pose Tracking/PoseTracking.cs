// Copyright (c) 2021 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// ATTENTION!: This code is for a tutorial.

using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using Mediapipe.Unity.CoordinateSystem;
using Mediapipe.Unity;

using Stopwatch = System.Diagnostics.Stopwatch;
using System.Collections.Generic;
using System;
using Mediapipe;

public class PoseTracking : MonoBehaviour
{

    [SerializeField] private TextAsset _configAsset;
    [SerializeField] private RawImage _screen;
    [SerializeField] private int _width;
    [SerializeField] private int _height;
    [SerializeField] private int _fps;
    [SerializeField] private PoseLandmarkListAnnotationController _poseLandmarksAnnotationController;

    private CalculatorGraph _graph;

    private WebCamTexture _webCamTexture;
    private Texture2D _inputTexture;
    private Color32[] _inputPixelData;
    private Texture2D _outputTexture;
    private Color32[] _outputPixelData;
    private UnityEngine.Rect _screenRect;

    private readonly Stopwatch _stopwatch = new Stopwatch();
    private OutputStream<ImageFramePacket, ImageFrame> _outputVideoStream;
    private OutputStream<NormalizedLandmarkListPacket, NormalizedLandmarkList> _poseLandmarksStream;

    private static bool _IsFirst = true;
    private Vector3 _bodyLandmarks;

    private IEnumerator Start()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            throw new Exception("Web Camera devices are not found");
        }


#if UNITY_EDITOR
        var webCamDevice = WebCamTexture.devices[0];
#else
      var webCamDevice = WebCamTexture.devices[1];
#endif

        _webCamTexture = new WebCamTexture(webCamDevice.name, _width, _height, _fps);
        _webCamTexture.Play();

        yield return new WaitUntil(() => _webCamTexture.width > 16);

        _screen.rectTransform.sizeDelta = new Vector2(_width, _height);

        _inputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
        _inputPixelData = new Color32[_width * _height];
        _outputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
        _outputPixelData = new Color32[_width * _height];

        _screen.texture = _outputTexture;

        if (_IsFirst)
        {
            AssetLoader.Provide(new StreamingAssetsResourceManager());
        }
        yield return AssetLoader.PrepareAssetAsync("pose_landmark_full.bytes", "pose_landmark_full.bytes", false);
        yield return AssetLoader.PrepareAssetAsync("pose_detection.bytes", "pose_detection.bytes", false);

        var config = CalculatorGraphConfig.Parser.ParseFromTextFormat(_configAsset.text);

        yield return GpuManager.Initialize();

        if (!GpuManager.IsInitialized)
        {
            throw new Exception("Failed to initialize GPU resources");
        }

        using (var validatedGraphConfig = new ValidatedGraphConfig())
        {
            validatedGraphConfig.Initialize(config).AssertOk();
            _graph = new CalculatorGraph(validatedGraphConfig.Config());
            _graph.SetGpuResources(GpuManager.GpuResources).AssertOk();
        }

        _outputVideoStream = new OutputStream<ImageFramePacket, ImageFrame>(_graph, "output_video");
        _poseLandmarksStream = new OutputStream<NormalizedLandmarkListPacket, NormalizedLandmarkList>(_graph, "pose_landmarks");
        _outputVideoStream.StartPolling().AssertOk();
        _poseLandmarksStream.StartPolling().AssertOk();
        _stopwatch.Start();

        _graph.StartRun().AssertOk();

        _screenRect = _screen.GetComponent<RectTransform>().rect;

        _IsFirst = false;

    }

    private void Update()
    {
        _inputTexture.SetPixels32(_webCamTexture.GetPixels32(_inputPixelData));
        var imageFrame = new ImageFrame(ImageFormat.Types.Format.Srgba, _width, _height, _width * 4, _inputTexture.GetRawTextureData<byte>());
        var currentTimestamp = _stopwatch.ElapsedTicks / (TimeSpan.TicksPerMillisecond / 1000);
        _graph.AddPacketToInputStream("input_video", new ImageFramePacket(imageFrame, new Timestamp(currentTimestamp))).AssertOk();


        if (_outputVideoStream.TryGetNext(out var outputVideo))
        {
            if (outputVideo.TryReadPixelData(_outputPixelData))
            {
                _outputTexture.SetPixels32(_outputPixelData);
                _outputTexture.Apply();
            }
        }
        if (_poseLandmarksStream.TryGetNext(out var poseLandmarks))
        {
            if (poseLandmarks != null)
            {
                for (int i = 1; i <= 32; i++)
                {
                    _bodyLandmarks = _screenRect.GetPoint(poseLandmarks.Landmark[i]);
                    Debug.Log(_bodyLandmarks);
                }
            }
        }
    }

    private void OnDestroy()
    {
        if (_webCamTexture != null)
        {
            _webCamTexture.Stop();
        }

        if (_graph != null)
        {
            _graph.CloseInputStream("input_video").AssertOk();
            _graph.WaitUntilDone().AssertOk();
            _graph.Dispose();
        }
        GpuManager.Shutdown();
    }
}
