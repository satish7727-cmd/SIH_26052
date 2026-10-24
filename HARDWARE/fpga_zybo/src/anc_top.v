module anc_top (
    input  wire sys_clk,
    input  wire reset_n,
    input  wire i2s_bclk,
    input  wire i2s_lrclk,
    input  wire i2s_sdata_in,
    output wire i2s_sdata_out,
    output wire led_impulse
);
    wire signed [23:0] mic_ref, mic_ep, mic_ef1, mic_ef2;
    wire frame_valid, fast_impulse;
    wire signed [23:0] anti_noise_y;

    i2s_tdm_receiver u_i2s_rx (
        .sys_clk(sys_clk), .reset_n(reset_n),
        .i2s_bclk(i2s_bclk), .i2s_lrclk(i2s_lrclk), .i2s_sdata_in(i2s_sdata_in),
        .mic_ref(mic_ref), .mic_ep(mic_ep), .mic_ef1(mic_ef1), .mic_ef2(mic_ef2),
        .frame_valid(frame_valid)
    );

    fast_feature_extractor u_fast_path (
        .clk(sys_clk), .reset(!reset_n),
        .audio_in(mic_ref), .sample_valid(frame_valid),
        .fast_impulse_detect(fast_impulse)
    );

    fxlms_filter u_fxlms (
        .clk(sys_clk), .reset(!reset_n),
        .sample_valid(frame_valid),
        .x_in(mic_ref), .e_in(mic_ep),
        .step_mu(32'h00000400),
        .y_out(anti_noise_y)
    );

    assign led_impulse = fast_impulse;
endmodule
