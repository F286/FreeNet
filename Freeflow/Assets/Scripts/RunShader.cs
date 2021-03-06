﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RunShader : MonoBehaviour {

    public ComputeShader shader;
    public MeshRenderer quad;
    public Texture2D source;

    public float[] filter;

    public void Update()
    {
        int kernelHandle = shader.FindKernel("CSMain");

        RenderTexture destination = new RenderTexture(256, 256, 24);
        //print(source.format);
        destination.enableRandomWrite = true;
        destination.Create();

        // Set filter buffer
        ComputeBuffer filterBuffer = new ComputeBuffer(filter.Length, 4);
        filterBuffer.SetData(filter);

        shader.SetTexture(kernelHandle, "Source", source);
        shader.SetTexture(kernelHandle, "Destination", destination);
        shader.SetBuffer(kernelHandle, "Filter", filterBuffer);
        shader.Dispatch(kernelHandle, 256 / 8, 256 / 8, 1);

        // Get filter buffer
        filterBuffer.GetData(filter);

        quad.material.mainTexture = destination;
    }
    
}
