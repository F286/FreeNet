using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RunShader : MonoBehaviour {

    public ComputeShader shader;
    public MeshRenderer renderer;

    public float[] filter;

    public void Update()
    {
        int kernelHandle = shader.FindKernel("CSMain");

        RenderTexture tex = new RenderTexture(256, 256, 24);
        tex.enableRandomWrite = true;
        tex.Create();

        // Set filter buffer
        ComputeBuffer filterBuffer = new ComputeBuffer(filter.Length, 4);
        filterBuffer.SetData(filter);

        shader.SetTexture(kernelHandle, "Result", tex);
        shader.SetBuffer(kernelHandle, "Filter", filterBuffer);
        shader.Dispatch(kernelHandle, 256 / 8, 256 / 8, 1);

        // Get filter buffer
        filterBuffer.GetData(filter);

        renderer.material.mainTexture = tex;
    }
    
}
